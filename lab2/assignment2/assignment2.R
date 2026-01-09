library(dplyr)
library(tidyr)
library(tree)
library(MLmetrics)
library(caret)
library(rpart)

#TASK 1 & 2

data_raw <-read.csv("bank-full.csv", stringsAsFactors = TRUE, sep=";")
data_raw <- data_raw%>%select(-duration)

#split data into three sections
set.seed(12345)
n <- dim(data_raw)[1]

train_idx <- sample(1:n, floor(n* 0.40)) 

data_train <- data_raw[train_idx,]

remaining_idx <- setdiff(1:n, train_idx)
test_idx <- sample(remaining_idx, floor(n*0.50))
data_test <- data_raw[test_idx,]

valid_idx <- setdiff(remaining_idx, test_idx)
data_valid <- data_raw[valid_idx,]

#data_train <- data_train%>%select(-y)
data_target <- data_raw$y

tree_1 <- tree(y ~ ., data=data_train)
tree_2 <- tree(y ~ ., data=data_train, control=tree.control(nrow(data_train),minsize=7000))
tree_3 <- tree(y ~ ., data=data_train, control=tree.control(nrow(data_train), mindev=0.0005))

#Missclassification rate on training data
summary(tree_1) #0.1048  
summary(tree_2) #0.1048  
summary(tree_3) #0.09362  

pred_1 <- predict(tree_1, newdata = data_valid, type="class") 
pred_2 <- predict(tree_2, newdata = data_valid, type="class")
pred_3 <- predict(tree_3, newdata = data_valid, type="class")

#missclassification on validation data
missclass_1 <- mean(pred_1 != data_valid$y)
missclass_2 <- mean(pred_2 != data_valid$y)
missclass_3 <- mean(pred_3 != data_valid$y)

print(missclass_1) #0.1156568
print(missclass_2) #0.1156568
print(missclass_3) #0.1141088

#TASK 3

train_score <- rep(0,50)
valid_score <- rep(0,50)

for (i in 2:50){
  tree_pruned <- prune.tree(tree_3, best=i)
  pred_valid <- predict(tree_pruned, newdata=data_valid, type="tree")
  
  train_score[i] <- deviance(tree_pruned)
  valid_score[i] <- deviance(pred_valid)
}

plot(train_score, type="b", col="red")
points(valid_score, type="b", cole="blue")

optimal_train <- min(train_score[2:50])
optimal_valid <- min(valid_score[2:50])

data.frame(
  train=optimal_train,
  valid=optimal_valid
)

opt_tree <- prune.tree(tree_3, best=optimal_valid)
plot(opt_tree)

#task 4

pred_test <- predict(opt_tree, newdata=data_test, type="class")

cm <- table(pred_test, data_test$y)
TP <- cm["yes","yes"]
TN <- cm["no","no"]
FP <- cm["no", "yes"]
FN <- cm["yes", "no"]

accuracy <- (TP+TN)/(TP+TN+FP+FN)
precision <- TP/(TP+FP)
recall <- TP / (TP + FN)

F1 <- 2*(precision * recall)/(precision+recall) #data is imbalanced and therefore F1 is more suitable

accuracy #this is high due to many correct TN classifications
precision #very low, the model is not very good at predicting the Positive class
recall #
F1 #low

# TASK 5
loss_matrix <- matrix(c(0,1,5,0), nrow=2, byrow=TRUE)
probabilities <- predict(opt_tree, newdata=data_test, type="vector")
losses <- probabilities %*% loss_matrix

best_i <- apply(losses, MARGIN = 1, FUN = which.min)
pred <- levels(data_test$y)[best_i]

cm <- table(pred, data_test$y)
TP <- cm["yes","yes"]
TN <- cm["no","no"]
FP <- cm["no", "yes"]
FN <- cm["yes", "no"]

accuracy <- (TP+TN)/(TP+TN+FP+FN)
precision <- TP/(TP+FP)
recall <- TP / (TP + FN)
F1 <- 2*(precision * recall)/(precision+recall)
accuracy
F1

#Accuracy decreased because of high punishment on FN, predictive power increased since F1 increased due to to higher
#penalty on FN is higher than FP to balance out the imbalanced dataset

#TASK 6

prob_tree <- predict(opt_tree, newdata=data_test, type="vector")

TPR <- rep(0, 19)
FPR <- rep(0, 19)
index <- 1
for(i in seq(from=0.05, to=0.95, by=0.05)){
  pred <- ifelse(prob_tree[,"yes"] > i, "yes", "no")
  cm <- table(pred, data_test$y)
  TP <- cm["yes","yes"]
  TN <- cm["no","no"]
  FP <- cm["no", "yes"]
  FN <- cm["yes", "no"]
  TPR[index] <- TP/(TP + FN)
  FPR[index] <- FP/(TN + FP)
  index <- index + 1
}

plot(FPR, TPR, pch=5, type="b")

#logistic regression classifier

log_model <- glm(y ~ ., data=data_train, family="binomial")
probs <- predict(log_model, newdata = data_test, type="response")

TPR <- rep(0, 19)
FPR <- rep(0, 19)
index <- 1
for(i in seq(from=0.05, to=0.95, by=0.05)){
  pred <- ifelse(probs> i, "yes", "no")
  cm <- table(pred, data_test$y)
  TP <- cm["yes","yes"]
  TN <- cm["no","no"]
  FP <- cm["no", "yes"]
  FN <- cm["yes", "no"]
  TPR[index] <- TP/(TP + FN)
  FPR[index] <- FP/(TN + FP)
  index <- index + 1
}

plot(FPR, TPR, pch=5, type="b")


