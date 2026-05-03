/*
 * Linear regression implementations (OLS, ridge, lasso).
 *
 * OLS / ridge use the normal equations and a Cholesky factorisation.
 * Lasso uses cyclic coordinate descent with soft-thresholding.
 *
 * Single translation unit, no external BLAS dependency, OpenMP-parallel
 * where it matters.
 */

#include "regression.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static float soft_threshold(float z, float lam) {
    if (z > lam) return z - lam;
    if (z < -lam) return z + lam;
    return 0.0f;
}

/* In-place Cholesky factorisation A -> L * L^T (lower-triangular L stored
 * over A's lower triangle).  Returns 0 on success, -1 if A is not SPD. */
static int cholesky(float *A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = A[i * n + j];
            for (int k = 0; k < j; ++k)
                sum -= A[i * n + k] * A[j * n + k];
            if (i == j) {
                if (sum <= 0.0f) return -1;
                A[i * n + i] = sqrtf(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
    return 0;
}

/* Solve L y = b (forward substitution) */
static void forward_sub(const float *L, const float *b, float *y, int n) {
    for (int i = 0; i < n; ++i) {
        float sum = b[i];
        for (int j = 0; j < i; ++j) sum -= L[i * n + j] * y[j];
        y[i] = sum / L[i * n + i];
    }
}

/* Solve L^T x = y (backward substitution) */
static void backward_sub(const float *L, const float *y, float *x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        float sum = y[i];
        for (int j = i + 1; j < n; ++j) sum -= L[j * n + i] * x[j];
        x[i] = sum / L[i * n + i];
    }
}

static void fit_closed_form(const float *X, const float *y,
                            int n, int d,
                            float ridge_lambda,
                            float *weights, float *bias) {
    int D = d + 1;
    float *A = (float *)calloc((size_t)D * D, sizeof(float));
    float *b = (float *)calloc((size_t)D, sizeof(float));
    if (!A || !b) { free(A); free(b); return; }

    /* X_aug = [X | 1].  Build A = X_aug^T X_aug and b = X_aug^T y. */
#pragma omp parallel for
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) {
            float acc = 0.0f;
            for (int s = 0; s < n; ++s) {
                float xi = (i < d) ? X[(size_t)s * d + i] : 1.0f;
                float xj = (j < d) ? X[(size_t)s * d + j] : 1.0f;
                acc += xi * xj;
            }
            A[(size_t)i * D + j] = acc;
            A[(size_t)j * D + i] = acc;
        }
        float acc = 0.0f;
        for (int s = 0; s < n; ++s) {
            float xi = (i < d) ? X[(size_t)s * d + i] : 1.0f;
            acc += xi * y[s];
        }
        b[i] = acc;
    }

    if (ridge_lambda > 0.0f) {
        for (int i = 0; i < d; ++i) A[(size_t)i * D + i] += ridge_lambda;
    }

    if (cholesky(A, D) != 0) {
        /* Diagonal jitter and retry */
        for (int i = 0; i < D; ++i) A[(size_t)i * D + i] += 1e-4f;
        cholesky(A, D);
    }
    float *yvec = (float *)malloc((size_t)D * sizeof(float));
    float *xvec = (float *)malloc((size_t)D * sizeof(float));
    forward_sub(A, b, yvec, D);
    backward_sub(A, yvec, xvec, D);

    for (int i = 0; i < d; ++i) weights[i] = xvec[i];
    *bias = xvec[d];

    free(A); free(b); free(yvec); free(xvec);
}

static void fit_lasso_cd(const float *X, const float *y,
                         int n, int d, float lam, int num_iter,
                         float *weights, float *bias) {
    /* Pre-compute column squared norms divided by n */
    float *col_sq = (float *)malloc((size_t)d * sizeof(float));
    for (int j = 0; j < d; ++j) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) {
            float v = X[(size_t)i * d + j];
            s += v * v;
        }
        col_sq[j] = s / (float)n;
        if (col_sq[j] < 1e-12f) col_sq[j] = 1e-12f;
    }
    for (int j = 0; j < d; ++j) weights[j] = 0.0f;
    float b = 0.0f;
    for (int i = 0; i < n; ++i) b += y[i];
    b /= (float)n;

    /* residual = y - X w - b */
    float *res = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        float pred = b;
        for (int j = 0; j < d; ++j) pred += X[(size_t)i * d + j] * weights[j];
        res[i] = y[i] - pred;
    }

    for (int it = 0; it < num_iter; ++it) {
        float max_change = 0.0f;
        for (int j = 0; j < d; ++j) {
            float old = weights[j];
            float rho = 0.0f;
            for (int i = 0; i < n; ++i)
                rho += X[(size_t)i * d + j] * (res[i] + X[(size_t)i * d + j] * old);
            rho /= (float)n;
            float new_w = soft_threshold(rho, lam) / col_sq[j];
            float ch = fabsf(new_w - old);
            if (ch > max_change) max_change = ch;
            weights[j] = new_w;
            float diff = old - new_w;
            for (int i = 0; i < n; ++i)
                res[i] += X[(size_t)i * d + j] * diff;
        }
        /* Update bias */
        float mean_res = 0.0f;
        for (int i = 0; i < n; ++i) mean_res += res[i];
        mean_res /= (float)n;
        for (int i = 0; i < n; ++i) res[i] -= mean_res;
        b += mean_res;
        if (max_change < 1e-6f) break;
    }
    *bias = b;
    free(res); free(col_sq);
}

void regression_fit(const float *X, const float *y,
                    int n, int d,
                    int regularizer, float lambda, int num_iter,
                    float *weights, float *bias) {
    if (regularizer == 2) {
        fit_lasso_cd(X, y, n, d, lambda, num_iter, weights, bias);
    } else {
        float ridge = (regularizer == 1) ? lambda : 0.0f;
        fit_closed_form(X, y, n, d, ridge, weights, bias);
    }
}

void regression_predict(const float *X, const float *w, float b,
                        int n, int d, float *y_hat) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float s = b;
        for (int j = 0; j < d; ++j) s += X[(size_t)i * d + j] * w[j];
        y_hat[i] = s;
    }
}
