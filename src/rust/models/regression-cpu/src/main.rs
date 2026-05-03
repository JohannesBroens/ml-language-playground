//! Linear / ridge / lasso regression with parallel inner loops.
//!
//! OLS and ridge use the normal equations and a Cholesky factorisation;
//! lasso uses cyclic coordinate descent with soft-thresholding.  The
//! synthetic-linear dataset generation matches the Python implementation
//! so the cross-language benchmark numbers stay comparable.

use std::env;
use std::time::Instant;

use rayon::prelude::*;

const DEFAULT_FEATURES: usize = 20;
const DEFAULT_INFORMATIVE: usize = 5;

// ---------- argument parsing ----------

fn arg_value<T: std::str::FromStr>(flag: &str, default: T) -> T {
    let mut iter = env::args();
    let _ = iter.next();
    let mut prev: Option<String> = None;
    for cur in iter {
        if let Some(p) = &prev {
            if p == flag {
                if let Ok(v) = cur.parse::<T>() {
                    return v;
                }
            }
        }
        prev = Some(cur);
    }
    default
}

fn arg_str(flag: &str, default: &str) -> String {
    let mut iter = env::args();
    let _ = iter.next();
    let mut prev: Option<String> = None;
    for cur in iter {
        if let Some(p) = &prev {
            if p == flag {
                return cur;
            }
        }
        prev = Some(cur);
    }
    default.to_string()
}

// ---------- a tiny xorshift RNG ----------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdead_beefu64 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f32) / ((1u64 << 53) as f32)
    }
    fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }
    fn randn(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

// ---------- synthetic-linear dataset ----------

fn generate_synthetic_linear(n: usize, d: usize, n_inf: usize, seed: u64)
    -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut x = vec![0.0f32; n * d];
    for v in x.iter_mut() { *v = rng.randn(); }
    let mut beta = vec![0.0f32; d];
    let mut filled = 0usize;
    while filled < n_inf {
        let idx = rng.range(d);
        if beta[idx] == 0.0 {
            beta[idx] = rng.uniform_range(-3.0, 3.0);
            filled += 1;
        }
    }
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..d { acc += x[i * d + j] * beta[j]; }
        acc += 0.5 * rng.randn();
        y[i] = acc;
    }
    (x, y)
}

fn normalize(x: &mut [f32], y: &mut [f32], n: usize, d: usize) {
    let mean: f32 = y.iter().copied().sum::<f32>() / n as f32;
    let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = (var + 1e-8).sqrt();
    for v in y.iter_mut() { *v = (*v - mean) / std; }
    for j in 0..d {
        let mut m = 0.0f32; let mut v = 0.0f32;
        for i in 0..n { m += x[i * d + j]; }
        m /= n as f32;
        for i in 0..n { v += (x[i * d + j] - m).powi(2); }
        let sd = (v / n as f32 + 1e-8).sqrt();
        for i in 0..n { x[i * d + j] = (x[i * d + j] - m) / sd; }
    }
}

fn split(x: &[f32], y: &[f32], n: usize, d: usize, seed: u64)
    -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
    let ntr = (n as f32 * 0.8) as usize;
    let nte = n - ntr;
    let mut xtr = vec![0.0f32; ntr * d];
    let mut xte = vec![0.0f32; nte * d];
    let mut ytr = vec![0.0f32; ntr];
    let mut yte = vec![0.0f32; nte];
    for i in 0..ntr {
        xtr[i * d..(i + 1) * d].copy_from_slice(&x[idx[i] * d..(idx[i] + 1) * d]);
        ytr[i] = y[idx[i]];
    }
    for i in 0..nte {
        let s = idx[ntr + i];
        xte[i * d..(i + 1) * d].copy_from_slice(&x[s * d..(s + 1) * d]);
        yte[i] = y[s];
    }
    (xtr, ytr, xte, yte)
}

// ---------- linear algebra: in-place Cholesky and triangular solves ----------

fn cholesky(a: &mut [f32], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j { sum -= a[i * n + k] * a[j * n + k]; }
            if i == j {
                if sum <= 0.0 { return false; }
                a[i * n + i] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }
    true
}

fn forward_sub(l: &[f32], b: &[f32], y: &mut [f32], n: usize) {
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i { sum -= l[i * n + j] * y[j]; }
        y[i] = sum / l[i * n + i];
    }
}

fn backward_sub(l: &[f32], y: &[f32], x: &mut [f32], n: usize) {
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n { sum -= l[j * n + i] * x[j]; }
        x[i] = sum / l[i * n + i];
    }
}

// ---------- fitting routines ----------

fn fit_closed_form(x: &[f32], y: &[f32], n: usize, d: usize, ridge: f32)
    -> (Vec<f32>, f32) {
    let big = d + 1;
    let mut a = vec![0.0f32; big * big];
    let mut b = vec![0.0f32; big];

    // Build symmetric system.  Parallel over rows.
    let rows: Vec<(Vec<f32>, f32)> = (0..big).into_par_iter().map(|i| {
        let mut row = vec![0.0f32; big];
        for j in i..big {
            let mut acc = 0.0f32;
            for s in 0..n {
                let xi = if i < d { x[s * d + i] } else { 1.0 };
                let xj = if j < d { x[s * d + j] } else { 1.0 };
                acc += xi * xj;
            }
            row[j] = acc;
        }
        let mut rhs = 0.0f32;
        for s in 0..n {
            let xi = if i < d { x[s * d + i] } else { 1.0 };
            rhs += xi * y[s];
        }
        (row, rhs)
    }).collect();

    for (i, (row, rhs)) in rows.into_iter().enumerate() {
        for j in i..big {
            a[i * big + j] = row[j];
            a[j * big + i] = row[j];
        }
        b[i] = rhs;
    }

    if ridge > 0.0 {
        for i in 0..d { a[i * big + i] += ridge; }
    }

    if !cholesky(&mut a, big) {
        for i in 0..big { a[i * big + i] += 1e-4; }
        cholesky(&mut a, big);
    }
    let mut yv = vec![0.0f32; big];
    let mut xv = vec![0.0f32; big];
    forward_sub(&a, &b, &mut yv, big);
    backward_sub(&a, &yv, &mut xv, big);
    let bias = xv[d];
    let weights = xv[..d].to_vec();
    (weights, bias)
}

fn fit_lasso(x: &[f32], y: &[f32], n: usize, d: usize, lam: f32, num_iter: usize)
    -> (Vec<f32>, f32) {
    let mut col_sq = vec![0.0f32; d];
    for j in 0..d {
        let mut s = 0.0f32;
        for i in 0..n {
            let v = x[i * d + j];
            s += v * v;
        }
        col_sq[j] = (s / n as f32).max(1e-12);
    }
    let mut w = vec![0.0f32; d];
    let mut b = y.iter().copied().sum::<f32>() / n as f32;
    let mut res = vec![0.0f32; n];
    for i in 0..n {
        let mut p = b;
        for j in 0..d { p += x[i * d + j] * w[j]; }
        res[i] = y[i] - p;
    }
    for _ in 0..num_iter {
        let mut max_change = 0.0f32;
        for j in 0..d {
            let old = w[j];
            let mut rho = 0.0f32;
            for i in 0..n {
                rho += x[i * d + j] * (res[i] + x[i * d + j] * old);
            }
            rho /= n as f32;
            let new_w = if rho > lam {
                (rho - lam) / col_sq[j]
            } else if rho < -lam {
                (rho + lam) / col_sq[j]
            } else {
                0.0
            };
            let ch = (new_w - old).abs();
            if ch > max_change { max_change = ch; }
            w[j] = new_w;
            let diff = old - new_w;
            for i in 0..n { res[i] += x[i * d + j] * diff; }
        }
        let mean_res: f32 = res.iter().copied().sum::<f32>() / n as f32;
        for v in res.iter_mut() { *v -= mean_res; }
        b += mean_res;
        if max_change < 1e-6 { break; }
    }
    (w, b)
}

fn predict(x: &[f32], w: &[f32], b: f32, n: usize, d: usize) -> Vec<f32> {
    (0..n).into_par_iter().map(|i| {
        let mut s = b;
        for j in 0..d { s += x[i * d + j] * w[j]; }
        s
    }).collect()
}

fn evaluate(xte: &[f32], yte: &[f32], w: &[f32], b: f32, nte: usize, d: usize)
    -> (f32, f32, f32) {
    let pred = predict(xte, w, b, nte, d);
    let mean: f32 = yte.iter().copied().sum::<f32>() / nte as f32;
    let mut sse = 0.0f32;
    let mut sst = 0.0f32;
    for i in 0..nte {
        let e = pred[i] - yte[i];
        sse += e * e;
        sst += (yte[i] - mean).powi(2);
    }
    let mse = sse / nte as f32;
    let rmse = mse.sqrt();
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
    (mse, rmse, r2)
}

fn main() {
    let n: usize = arg_value("--num-samples", 4096);
    let d: usize = DEFAULT_FEATURES;
    let n_inf: usize = DEFAULT_INFORMATIVE;
    let num_iter: usize = arg_value("--epochs", 200);
    let lambda: f32 = arg_value("--lambda-reg", 0.1f32);
    let reg = arg_str("--regularizer", "none");
    let reg_id = match reg.as_str() {
        "l2" => 1,
        "l1" => 2,
        _ => 0,
    };

    let (mut x, mut y) = generate_synthetic_linear(n, d, n_inf, 1234);
    normalize(&mut x, &mut y, n, d);
    let (xtr, ytr, xte, yte) = split(&x, &y, n, d, 5678);
    let ntr = ytr.len();
    let nte = yte.len();

    println!("Dataset: synthetic-linear  ({} samples, {} features, {} informative)",
             n, d, n_inf);
    println!("Train: {} | Test: {}", ntr, nte);
    println!("Model: regression (regularizer={}, solver={})",
             reg, if reg_id == 2 { "coord-descent" } else { "closed-form" });

    let t0 = Instant::now();
    let (w, b) = if reg_id == 2 {
        fit_lasso(&xtr, &ytr, ntr, d, lambda, num_iter)
    } else {
        let ridge = if reg_id == 1 { lambda } else { 0.0 };
        fit_closed_form(&xtr, &ytr, ntr, d, ridge)
    };
    let t_train = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let (mse, rmse, r2) = evaluate(&xte, &yte, &w, b, nte, d);
    let t_eval = t1.elapsed().as_secs_f64();

    let nonzero = w.iter().filter(|v| v.abs() > 1e-6).count();
    let throughput = (ntr as f64 * num_iter as f64) / t_train.max(1e-9);

    println!();
    println!("=== Results on synthetic-linear ===");
    println!("Test Loss:     {:.4}", mse);
    println!("Test Accuracy: {:.2}%", r2 * 100.0);
    println!("Test RMSE:     {:.4}", rmse);
    println!("Non-zero w:    {}/{}", nonzero, d);
    println!("Train time:    {:.3} s", t_train);
    println!("Eval time:     {:.3} s", t_eval);
    println!("Throughput:    {:.0} samples/s", throughput);
}
