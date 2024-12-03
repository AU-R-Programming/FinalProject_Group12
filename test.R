X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2)
y <- c(0, 1, 1, 0)
beta <- c(0.5, -0.2)
initial_values(X, y)
logistic_loss(beta, X, y)
estimate_coefficients(X, y)
bootstrap_ci(X, y, alpha = 0.05, n_bootstraps = 100)

X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2) # Predictors
beta <- c(-1, 0.5) # Coefficients
predict_class(X, beta, cutoff = 0.5)

y <- c(1, 0, 1, 1, 0, 0, 1)
y_pred <- c(1, 0, 1, 0, 0, 1, 1)
confusion_matrix(y, y_pred)
