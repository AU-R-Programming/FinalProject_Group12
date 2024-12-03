
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

  # Check if the first column is not all ones (indicating no intercept column)
  if (!all(X[, 1] == 1)) {
    X <- cbind(1, X)  # Add intercept column
  }

  beta_0 <- solve(t(X) %*% X) %*% t(X) %*% y
  return(as.numeric(beta_0))
}





#' @title Logistic Loss Function
#'
#' @description Computes the logistic loss for a given set of coefficients, predictor variables, and binary response variables.
#' The logistic loss is the negative log-likelihood of the logistic regression model, used as an objective function for optimization.
#'
#' @param beta A numeric vector of coefficients for the logistic regression model.
#' @param X A numeric matrix of predictor variables.
#' @param y A numeric vector of binary response variables (0 or 1).
#' @return A numeric scalar representing the logistic loss value.
#' @examples
#' beta <- c(0.5, -0.2) # Example coefficients
#' X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2) # Predictors with intercept column
#' y <- c(0, 1, 1, 0) # Binary responses
#' logistic_loss(beta, X, y)
#' @export
logistic_loss <- function(beta, X, y) {
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- length(y)
  p <- 1 / (1 + exp(-X %*% beta))
  -sum(y * log(p) + (1 - y) * log(1 - p))
}






#' @title Estimate Coefficients for Logistic Regression
#'
#' @description Estimates the coefficients for a logistic regression model by minimizing the logistic loss function
#' using the Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization method. Initial coefficient estimates are generated
#' using a least-squares approach.
#'
#' @param X A numeric matrix of predictor variables.
#' @param y A numeric vector of binary response variables (0 or 1).
#' @return A numeric vector of estimated coefficients for the logistic regression model.
#' @examples
#' X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2) # Predictors
#' y <- c(0, 1, 1, 0) # Binary responses
#' estimate_coefficients(X, y)
#' @export
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



#' @title Bootstrap Confidence Intervals for Logistic Regression Coefficients
#'
#' @description Computes bootstrap confidence intervals for logistic regression coefficients by resampling the data
#' and re-estimating the coefficients multiple times.
#'
#' @param X A numeric matrix of predictor variables.
#' @param y A numeric vector of binary response variables (0 or 1).
#' @param alpha A numeric value specifying the significance level for the confidence intervals (default is 0.05).
#' @param n_bootstraps An integer specifying the number of bootstrap resamples to perform (default is 20).
#' @return A list with two elements:
#' \describe{
#'   \item{lower}{A numeric vector of the lower bounds of the confidence intervals for each coefficient.}
#'   \item{upper}{A numeric vector of the upper bounds of the confidence intervals for each coefficient.}
#' }
#' @examples
#' X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2) # Predictors
#' y <- c(0, 1, 1, 0) # Binary responses
#' bootstrap_ci(X, y, alpha = 0.05, n_bootstraps = 100)
#' @export
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




#' @title Predict Binary Class Labels
#'
#' @description Predicts binary class labels based on logistic regression coefficients and a specified cutoff.
#'
#' @param X A numeric matrix of predictor variables.
#' @param beta A numeric vector of logistic regression coefficients.
#' @param cutoff A numeric value between 0 and 1 specifying the classification threshold (default is 0.5).
#' @return A numeric vector of predicted class labels (0 or 1).
#' @examples
#' X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2) # Predictors
#' beta <- c(-1, 0.5) # Coefficients
#' predict_class(X, beta, cutoff = 0.5)
#' @export
predict_class <- function(X, beta, cutoff = 0.5) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(ifelse(p > cutoff, 1, 0))
}




#' @title Confusion Matrix and Classification Metrics
#'
#' @description Computes a confusion matrix and associated classification metrics for evaluating the performance of binary classification models.
#'
#' @param y A numeric vector of actual binary response variables (0 or 1).
#' @param y_pred A numeric vector of predicted binary classifications (0 or 1).
#' @return A list containing:
#' \item{ConfusionMatrix}{A 2x2 matrix representing the confusion matrix, with rows corresponding to predicted classes and columns to actual classes.}
#' \item{Metrics}{A list of classification metrics, including:
#' \itemize{
#'   \item Prevalence: Proportion of positive cases in the actual data.
#'   \item Accuracy: Overall proportion of correctly classified observations.
#'   \item Sensitivity: True positive rate or recall.
#'   \item Specificity: True negative rate.
#'   \item FDR: False discovery rate.
#'   \item DOR: Diagnostic odds ratio.
#' }}
#' @examples
#' # Example actual and predicted values
#' y <- c(1, 0, 1, 1, 0, 0, 1)
#' y_pred <- c(1, 0, 1, 0, 0, 1, 1)
#' confusion_matrix(y, y_pred)
#' @export
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



