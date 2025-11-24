  
#ASSIGNMENT 2  

parkinsons = read.csv("parkinsons.csv", header=TRUE)
  
  n=dim(parkinsons)[1]
  set.seed(12345) 
  id=sample(1:n, floor(n*0.6)) 
  train=parkinsons[id,] 
  test=parkinsons[-id,] 
  
  library(dplyr)
  
  df=train%>%select(motor_UPDRS, Jitter...:PPE)
  train_df <- train %>% select(motor_UPDRS, Jitter...:PPE)
  test_df  <- test  %>% select(motor_UPDRS, Jitter...:PPE)
  
  # scale data
  library(caret)
  
  scaler <- preProcess(df)
  train_scaled <- predict(scaler, train_df)
  test_scaled <- predict(scaler, test_df)
  
  # Select features
  train_df <- train_scaled %>% select(motor_UPDRS, Jitter...:PPE)
  test_df  <- test_scaled  %>% select(motor_UPDRS, Jitter...:PPE)
  
  
  model=lm(motor_UPDRS~., train_scaled) 
  
  #Calculate prediction and MSE on the training data set
  Preds_train=predict(model, newdata=train_scaled)
  MSE_train=mean((train_df$motor_UPDRS-Preds_train)^2)
  
  #Calculate prediction and MSE on the test data set
  Preds_test=predict(model, newdata=test_scaled)
  MSE_test=mean((test_df$motor_UPDRS-Preds_test)^2)
  
  # 
  
  # Correct X for test set
  x <- model.matrix(motor_UPDRS ~ ., data = test_scaled)
  y <- test$motor_UPDRS
  
  loglike <- function(theta, sigma){
    n = length(y)
    return (-n/2 * log(2*pi*sigma^2) - 1/(2*sigma^2) * sum((y - x %*% theta)^2))
  }
  
  ridge <- function(theta, sigma,lambda){
    penalty_value <- lambda * sum(theta[-1]^2)
    return(-loglike(theta, sigma) + penalty_value)
  }
  
  # theta <- coef(model)
  # sigma <- sigma(model)
  # loglike(theta, sigma(model))
  # ridge(theta, sigma, 1)
  
  ridge_opt <- function(lambda){
    theta_init <- coef(model)
    sigma_init <- sigma(model)
    params_init <- c(theta_init, sigma_init)
    
    ridge_wrapper <- function(parameter_vector){
      n_theta <- length(theta_init)
      theta <- parameter_vector[1:n_theta]
      sigma <- parameter_vector[n_theta + 1]
      
      return(ridge(theta, sigma, lambda))
    }
      
    result <- optim(par=params_init, fn=ridge_wrapper, method="BFGS")
    return (result)
  }
  
  DF <- function(X, lambda){
    XtX <- t(X) %*% X
    eig_vals <- eigen(XtX, symmetric = TRUE)$values
    df <- sum(eig_vals / (eig_vals + lambda))
    return(df)
  }
  
  # compute optimal Ridge coefficients for different lambda values
  ridge_1    <- ridge_opt(lambda = 1)
  ridge_100  <- ridge_opt(lambda = 100)
  ridge_1000 <- ridge_opt(lambda = 1000)
  
  # Prepare feature matrices
  X_train <- as.matrix(train_scaled %>% select(Jitter...:PPE))
  X_test  <- as.matrix(test_scaled  %>% select(Jitter...:PPE))
  
  # Extract only the theta coefficients (exclude sigma)
  theta_1    <- as.matrix(ridge_1$par[-length(ridge_1$par)])
  theta_100  <- as.matrix(ridge_100$par[-length(ridge_100$par)])
  theta_1000 <- as.matrix(ridge_1000$par[-length(ridge_1000$par)])
  
  # Predictions
  
  # remove intercept from coefficients
  theta_1_no_intercept    <- theta_1[-1]
  theta_100_no_intercept  <- theta_100[-1]
  theta_1000_no_intercept <- theta_1000[-1]
  
  # prdictions for training and test sets
  y_train_pred_1    <- X_train %*% theta_1_no_intercept
  y_test_pred_1     <- X_test  %*% theta_1_no_intercept
  
  y_train_pred_100  <- X_train %*% theta_100_no_intercept
  y_test_pred_100   <- X_test  %*% theta_100_no_intercept
  
  y_train_pred_1000 <- X_train %*% theta_1000_no_intercept
  y_test_pred_1000  <- X_test  %*% theta_1000_no_intercept
  
  # Compute training and test MSE
  mse_train_1    <- mean((train_scaled$motor_UPDRS - y_train_pred_1)^2)
  mse_test_1     <- mean((test_scaled$motor_UPDRS  - y_test_pred_1)^2)
  
  mse_train_100  <- mean((train_scaled$motor_UPDRS - y_train_pred_100)^2)
  mse_test_100   <- mean((test_scaled$motor_UPDRS  - y_test_pred_100)^2)
  
  mse_train_1000 <- mean((train_scaled$motor_UPDRS - y_train_pred_1000)^2)
  mse_test_1000  <- mean((test_scaled$motor_UPDRS  - y_test_pred_1000)^2)
  
  # Compute degrees of freedom for each lambda
  df_1    <- DF(X_train, lambda = 1)
  df_100  <- DF(X_train, lambda = 100)
  df_1000 <- DF(X_train, lambda = 1000)
  
  
