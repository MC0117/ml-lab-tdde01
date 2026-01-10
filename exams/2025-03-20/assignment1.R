#############1#############

set.seed("12345")

raw_data <- read.csv("wdbc.csv")

vars_to_remove <- c("diagnosis", "area_mean")
data_clean <- raw_data[, !(names(raw_data) %in% vars_to_remove)]

data_scale <- as.data.frame(scale(data_clean))

library(glmnet)

data_clean <- data_clean[, colSums(is.na(data_clean))==0]
x_train <- as.matrix(data_clean)
y_train <- raw_data$area_mean
cv_lasso <- cv.glmnet(x_train, y_train, alpha=1, family="gaussian")


plot(cv_lasso)
opt_lambda <- cv_lasso$lambda.min
coef_optimal <- coef(cv_lasso, s=exp(-2))

vars_num <- sum(coef_optimal != 0) - 1 
print(vars_num)


#############2#############

set.seed("12345")

raw_data <- read.csv("wdbc.csv")

#vars_to_remove <- c("diagnosis")
#data_clean <- raw_data[, !(names(raw_data) %in% vars_to_remove)]

data_clean <- raw_data[, colSums(is.na(raw_data))==0]
data_clean$diagnosis <- ifelse(data_clean$diagnosis=="M", 1, 0)

n <- dim(data_clean)[1]

id_train <- sample(1:n, floor(n * 0.50))
train_data <- data_clean[id_train, ]
test_data <- data_clean[-id_train, ]

#data_clean$diagnosis <- as.factor(data_clean$diagnosis)
model_glm <- glm(diagnosis ~ ., family="binomial", data=train_data)

probs_training <- predict(model_glm, type="response", data=train_data)
probs_testing <- predict(model_glm, type="response", newdata=test_data) #viktigt new data

r <- 0.04761905

classified_training <- ifelse(probs_training > r, 1,0)
classified_testing <- ifelse(probs_testing > r, 1,0)

actuals_training <- train_data$diagnosis

MSE_train <- mean(classified_training != actuals_training)

actuals_testing <- test_data$diagnosis

MSE_test <- mean(classified_testing != actuals_testing)

print(MSE_train)
print(MSE_test)

#############3#############

y_hat <- function(x){ifelse(x < 0, "B", "M")}

hinge_loss <- function(w,X,y){
  
  #calculate scores
  scores <- X %*% w
  #calculate margin
  margins <- y * scores
  losses <- pmax(0, 1-margins)
  return(sum(losses))
  }

opt_hinge <- function(X, y){
  w_init <- rep(0, ncol(X))
  
  opt_cost <- optim(par = w_init,
                    fn = hinge_loss,
                    X=X,
                    y=y,
                    method="BFGS"
                    )  
  return(opt_cost)
}
library(dplyr)


X_train <- as.matrix(train_data%>%select(-diagnosis))
X_train <- cbind(1, X_train) #for intercept
y_train <- train_data$diagnosis

X_test <- test_data%>%select(-diagnosis)
X_test <- as.matrix(cbind(1, X_test)) #for intercept
y_test <- test_data$diagnosis

y_train <- ifelse(y_train == 1, 1, -1)
y_test  <- ifelse(y_test  == 1, 1, -1)


opt_result <- opt_hinge(X=X_train,y=y_train)
opt_w <- opt_result$par
opt_result$convergence

pred_train <- sign(X_train %*% opt_w)
pred_test <- sign(X_test %*% opt_w)

missclass_train <- mean(pred_train != y_train)
missclass_test <- mean(pred_test != y_test)

missclass_train
missclass_test

loss_train <- hinge_loss(opt_w, X_train, y_train)
loss_test <- hinge_loss(opt_w, X_test, y_test)
loss_train
loss_test

#Really bad here, classifies all as postitive class or negative, not sure if this is reasonable or not

