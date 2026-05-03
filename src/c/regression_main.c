/*
 * Linear / ridge / lasso regression CLI.
 *
 * Generates the same synthetic-linear dataset as the Python implementation
 * (20 features, ~5 informative, ~N(0, 1) inputs, weighted target with
 * Gaussian noise) so the cross-language benchmark numbers stay comparable.
 *
 * Output follows the standardised benchmark format:
 *   Test Loss / Test Accuracy / Train time / Eval time / Throughput
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "models/regression/regression.h"

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Box-Muller for standard normal */
static float randn(unsigned int *state) {
    float u1 = ((float)rand_r(state) + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand_r(state) + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void generate_synthetic_linear(int n, int d, int n_informative,
                                       float **X_out, float **y_out,
                                       float **beta_out, unsigned int seed) {
    float *X = (float *)malloc((size_t)n * d * sizeof(float));
    float *y = (float *)malloc((size_t)n * sizeof(float));
    float *beta = (float *)calloc((size_t)d, sizeof(float));
    unsigned int s = seed;
    /* Inputs */
    for (int i = 0; i < n * d; ++i) X[i] = randn(&s);
    /* Choose n_informative coefficients, set them ~ U(-3, 3) */
    for (int k = 0; k < n_informative; ++k) {
        int idx = rand_r(&s) % d;
        if (beta[idx] != 0.0f) { --k; continue; }
        beta[idx] = ((float)rand_r(&s) / RAND_MAX) * 6.0f - 3.0f;
    }
    for (int i = 0; i < n; ++i) {
        float yv = 0.0f;
        for (int j = 0; j < d; ++j) yv += X[(size_t)i * d + j] * beta[j];
        yv += 0.5f * randn(&s);
        y[i] = yv;
    }
    *X_out = X; *y_out = y;
    if (beta_out) *beta_out = beta; else free(beta);
}

static void normalize(float *X, float *y, int n, int d) {
    float ymean = 0.0f;
    for (int i = 0; i < n; ++i) ymean += y[i];
    ymean /= (float)n;
    float yvar = 0.0f;
    for (int i = 0; i < n; ++i) yvar += (y[i] - ymean) * (y[i] - ymean);
    float ystd = sqrtf(yvar / (float)n + 1e-8f);
    for (int i = 0; i < n; ++i) y[i] = (y[i] - ymean) / ystd;
    for (int j = 0; j < d; ++j) {
        float m = 0.0f;
        for (int i = 0; i < n; ++i) m += X[(size_t)i * d + j];
        m /= (float)n;
        float v = 0.0f;
        for (int i = 0; i < n; ++i) v += (X[(size_t)i * d + j] - m) * (X[(size_t)i * d + j] - m);
        float sd = sqrtf(v / (float)n + 1e-8f);
        for (int i = 0; i < n; ++i)
            X[(size_t)i * d + j] = (X[(size_t)i * d + j] - m) / sd;
    }
}

static void train_test_split(float *X, float *y, int n, int d,
                             float **Xtr, float **ytr,
                             float **Xte, float **yte,
                             int *ntr_out, int *nte_out,
                             unsigned int seed) {
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; ++i) idx[i] = i;
    unsigned int s = seed;
    for (int i = n - 1; i > 0; --i) {
        int j = rand_r(&s) % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
    int ntr = (int)(0.8f * n);
    int nte = n - ntr;
    *Xtr = (float *)malloc((size_t)ntr * d * sizeof(float));
    *Xte = (float *)malloc((size_t)nte * d * sizeof(float));
    *ytr = (float *)malloc((size_t)ntr * sizeof(float));
    *yte = (float *)malloc((size_t)nte * sizeof(float));
    for (int i = 0; i < ntr; ++i) {
        memcpy(*Xtr + (size_t)i * d, X + (size_t)idx[i] * d, d * sizeof(float));
        (*ytr)[i] = y[idx[i]];
    }
    for (int i = 0; i < nte; ++i) {
        memcpy(*Xte + (size_t)i * d, X + (size_t)idx[ntr + i] * d, d * sizeof(float));
        (*yte)[i] = y[idx[ntr + i]];
    }
    *ntr_out = ntr; *nte_out = nte;
    free(idx);
}

static void evaluate(const float *Xte, const float *yte, int nte, int d,
                     const float *w, float b,
                     float *mse_out, float *rmse_out, float *r2_out) {
    float *pred = (float *)malloc((size_t)nte * sizeof(float));
    regression_predict(Xte, w, b, nte, d, pred);
    float ymean = 0.0f;
    for (int i = 0; i < nte; ++i) ymean += yte[i];
    ymean /= (float)nte;
    float sse = 0.0f, sst = 0.0f;
    for (int i = 0; i < nte; ++i) {
        float e = pred[i] - yte[i];
        sse += e * e;
        sst += (yte[i] - ymean) * (yte[i] - ymean);
    }
    float mse = sse / (float)nte;
    *mse_out = mse;
    *rmse_out = sqrtf(mse);
    *r2_out = (sst > 0.0f) ? (1.0f - sse / sst) : 0.0f;
    free(pred);
}

static int parse_int_arg(int argc, char **argv, const char *flag, int dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
    return dflt;
}
static float parse_float_arg(int argc, char **argv, const char *flag, float dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return (float)atof(argv[i + 1]);
    return dflt;
}
static const char *parse_str_arg(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}

int main(int argc, char **argv) {
    int n = parse_int_arg(argc, argv, "--num-samples", 4096);
    if (n <= 0) n = 4096;
    int d = 20;
    int n_inf = 5;
    int num_iter = parse_int_arg(argc, argv, "--epochs", 200);
    float lambda = parse_float_arg(argc, argv, "--lambda-reg", 0.1f);
    const char *reg = parse_str_arg(argc, argv, "--regularizer", "none");
    int reg_id = 0;
    if (strcmp(reg, "l2") == 0) reg_id = 1;
    else if (strcmp(reg, "l1") == 0) reg_id = 2;

    float *X, *y, *beta;
    generate_synthetic_linear(n, d, n_inf, &X, &y, &beta, 1234u);
    normalize(X, y, n, d);
    float *Xtr, *Xte, *ytr, *yte;
    int ntr, nte;
    train_test_split(X, y, n, d, &Xtr, &ytr, &Xte, &yte, &ntr, &nte, 5678u);

    printf("Dataset: synthetic-linear  (%d samples, %d features, %d informative)\n",
           n, d, n_inf);
    printf("Train: %d | Test: %d\n", ntr, nte);
    printf("Model: regression (regularizer=%s, solver=%s)\n",
           reg, reg_id == 2 ? "coord-descent" : "closed-form");

    float *w = (float *)calloc((size_t)d, sizeof(float));
    float bias = 0.0f;
    double t0 = monotonic_seconds();
    regression_fit(Xtr, ytr, ntr, d, reg_id, lambda, num_iter, w, &bias);
    double t_train = monotonic_seconds() - t0;

    float mse, rmse, r2;
    double t1 = monotonic_seconds();
    evaluate(Xte, yte, nte, d, w, bias, &mse, &rmse, &r2);
    double t_eval = monotonic_seconds() - t1;

    int nonzero = 0;
    for (int j = 0; j < d; ++j) if (fabsf(w[j]) > 1e-6f) nonzero++;
    double throughput = (double)ntr * (double)num_iter / fmax(t_train, 1e-9);

    printf("\n=== Results on synthetic-linear ===\n");
    printf("Test Loss:     %.4f\n", mse);
    printf("Test Accuracy: %.2f%%\n", r2 * 100.0f);
    printf("Test RMSE:     %.4f\n", rmse);
    printf("Non-zero w:    %d/%d\n", nonzero, d);
    printf("Train time:    %.3f s\n", t_train);
    printf("Eval time:     %.3f s\n", t_eval);
    printf("Throughput:    %.0f samples/s\n", throughput);

    free(X); free(y); free(beta);
    free(Xtr); free(Xte); free(ytr); free(yte); free(w);
    return 0;
}
