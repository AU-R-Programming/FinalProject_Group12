
#' @title Initial Values for Logistic Regression
#'
#' @description Computes initial values for logistic regression coefficients using the least-squares formula.
#'
#' @param X A numeric matrix of predictor variables.
#' @param y A numeric vector of binary response variables (0 or 1).
#' @return A numeric vector of initial coefficient estimates.
#' @examples
#' X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2)
#' y <- c(0, 1, 1, 0)
#' initial_values(X, y)
#' @export
initial_values <- function(X, y) {
  X <- as.matrix(X)
  y <- as.vector(y)
  beta_0 <- solve(t(X) %*% X) %*% t(X) %*% y
  return(as.numeric(beta_0))
}


logistic_loss <- function(beta, X, y) {
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- length(y)
  p <- 1 / (1 + exp(-X %*% beta))
  -sum(y * log(p) + (1 - y) * log(1 - p))
}

estimate_coefficients <- function(X, y) {
  initial <- initial_values(X, y)
  result <- optim(
    par = initial,
    fn = logistic_loss,
    X = X,
    y = y,
    method = "BFGS"
  )
  return(result$par)
}

bootstrap_ci <- function(X, y, alpha = 0.05, n_bootstraps = 20) {
  n <- nrow(X)
  beta_samples <- replicate(n_bootstraps, {
    idx <- sample(1:n, size = n, replace = TRUE)
    X_boot <- X[idx, , drop = FALSE]
    y_boot <- y[idx]
    estimate_coefficients(X_boot, y_boot)
  })

  ci_lower <- apply(beta_samples, 1, quantile, probs = alpha / 2)
  ci_upper <- apply(beta_samples, 1, quantile, probs = 1 - alpha / 2)

  return(list(lower = ci_lower, upper = ci_upper))
}

predict_class <- function(X, beta, cutoff = 0.5) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(ifelse(p > cutoff, 1, 0))
}

confusion_matrix <- function(y, y_pred) {
  tp <- sum(y == 1 & y_pred == 1)
  tn <- sum(y == 0 & y_pred == 0)
  fp <- sum(y == 0 & y_pred == 1)
  fn <- sum(y == 1 & y_pred == 0)

  prevalence <- mean(y)
  accuracy <- (tp + tn) / length(y)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  fdr <- fp / (fp + tp)
  dor <- (sensitivity / (1 - specificity)) / ((1 - sensitivity) / specificity)

  return(list(
    ConfusionMatrix = matrix(c(tp, fp, fn, tn), 2, 2,
                             dimnames = list(c("Predicted:1", "Predicted:0"),
                                             c("Actual:1", "Actual:0"))),
    Metrics = list(
      Prevalence = prevalence,
      Accuracy = accuracy,
      Sensitivity = sensitivity,
      Specificity = specificity,
      FDR = fdr,
      DOR = dor
    )
  ))
}



