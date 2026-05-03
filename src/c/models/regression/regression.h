#ifndef REGRESSION_H
#define REGRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Linear regression with optional L1/L2 regularisation.
 *
 *   regularizer = 0  -> ordinary least squares (closed-form)
 *   regularizer = 1  -> ridge regression       (closed-form, weights only)
 *   regularizer = 2  -> lasso                  (coordinate descent)
 *
 * Weights are stored as a single flat vector of length num_features; the
 * intercept is returned separately.
 */
void regression_fit(const float *X, const float *y,
                    int n, int d,
                    int regularizer, float lambda,
                    int num_iter,
                    float *weights, float *bias);

/* Predict y_hat = X w + b for a batch */
void regression_predict(const float *X, const float *w, float b,
                        int n, int d, float *y_hat);

#ifdef __cplusplus
}
#endif

#endif /* REGRESSION_H */
