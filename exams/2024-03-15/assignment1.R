#TASK 1

data <- read.csv("women.csv")

set.seed(12345)
library(dplyr)
pc_data <- data %>% select(-Death)

pca_model <- prcomp(pc_data, scale.=TRUE)

summary(pca_model) #48.80%

pca_model$rotation[, "PC1"]

#-0.65177758*AGE -0.01864922*TumorSize -0.45472974*BloodPressure -0.26493200*Glucose-0.54577476*Cholestrol

df <- data.frame(
  PC1=pca_model$x[,"PC1"],
  PC2=pca_model$x[,"PC2"]
)


pca_model$x

library(ggplot2)

ggplot(data, aes(x = df$PC1, y = df$PC2, color = ifelse(data$Death, "red", "blue"))) +
  geom_point() +
  labs(title = "PC1 vs PC2",
       y = "PC2",
       x = "PC1") +
  theme_minimal()

#TASK 2

set.seed(12345)

n <- nrow(data)
id_train <- sample(1:n, floor(n * 0.50))
tr <- data[id_train, ]
ts <- data[-id_train, ]

observations_count <- seq(100,2500, by = 100) 
tree_res_tr <- rep(0, length(observations_count))
tree_res_ts <- rep(0, length(observations_count))
knn_res_tr <- rep(0, length(observations_count))
knn_res_ts <- rep(0, length(observations_count))

library(tree)
library(kknn)

for(i in 1:length(observations_count)){
  train <- tr[1:observations_count[i], ]
  tree_model <- tree(Cholestrol~., data=train)
  tree_model_pruned <- prune.tree(tree_model, best=10)
  tree_pred_tr <- predict(tree_model_pruned, newdata=train)
  tree_pred_ts <- predict(tree_model_pruned, newdata=ts)
  
  tree_res_tr[i] <- mean((train$Cholestrol - tree_pred_tr)^2)
  tree_res_ts[i] <- mean((ts$Cholestrol - tree_pred_ts)^2)
  
  knn_model <- kknn(Cholestrol ~ ., train = train, test = train, k = 10, kernel = "rectangular")
  knn_pred_tr <- knn_model$fitted.values
  
  knn_pred <- predict(knn_model, newdata=ts)
  
  knn_model <- kknn(Cholestrol ~ ., train = train, test = ts, k = 10, kernel = "rectangular")
  knn_pred_ts <- knn_model$fitted.values
  
  knn_res_tr[i] <- mean((train$Cholestrol - knn_pred_tr)^2)
  knn_res_ts[i] <- mean((ts$Cholestrol - knn_pred_ts)^2)
    
  
}
plot(tree_model_pruned)

plot(observations_count, tree_res_tr, type='', col='blue', ylim= range(c(knn_res_tr, knn_res_ts)))
points(observations_count, tree_res_ts, type='l', col='red')
legend("topright",
       legend=c("Training error", "Testing error"),
       col=c("blue", "red"), lty=1, pch=19)


y_lims <- range(c(knn_res_tr, knn_res_ts))

plot(observations_count, knn_res_tr, type='l', col='blue', ylim = range(c(knn_res_tr, knn_res_ts)))
points(observations_count, knn_res_ts, type='l', col='red')
legend("topright",
       legend=c("Training error", "Testing error"),
       col=c("blue", "red"), lty=1, pch=19)



#TASK 3 


lin_model <- lm(Death ~ ., data=tr)

print(summary(lin_model))

pred_tr <- predict(lin_model, tr)
pred_ts <- predict(lin_model, ts)

loglikelihood <- function(theta, sigma, X, y){
  n <- length(y)
  y_pred <- X %*% theta
  
  sse <- sum((y - y_pred)^2)
}
set.seed(12345)

library(glmnet)
library(dplyr)

tr <- data[id_train, ]
ts <- data[-id_train, ]

target_tr <- tr$Death
target_ts <- ts$Death

tr <- tr %>%select(-Death)
cv_glm <- cv.glmnet(as.matrix(tr), as.matrix(target_tr), alpha = 0, family = "binomial")
cv_glm #optimal lambda 0.0106
pred_ts <- predict(cv_glm, newx=as.matrix(ts[-6]), a="lambda.min", type="response")
# theorectical threshold FP/(FP+FN) = 1/11

pred_ts1 <- ifelse(pred_ts > 1/11, 1,0)

cm <- table(target_ts,pred_ts)

miss <- mean(pred_ts == target_ts)



