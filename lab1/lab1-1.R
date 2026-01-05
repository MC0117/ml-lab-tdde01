#TASK 1

digits <- read.csv("optdigits.csv", header=FALSE)
digits[,65] <- as.factor(digits[,65])

n <- dim(digits)[1]
set.seed(12345)
idx=sample(1:n)

#separate into training, testing and validation dataset
train_idx <- idx[1:floor(0.5*n)]
test_idx <- idx[(floor(0.5*n) + 1):floor(0.75*n)]
validation_idx <- idx[(floor(0.75*n) + 1):n]

train_df <- digits[train_idx,]
test_df <- digits[test_idx,]
validation_df <- digits[validation_idx,]

#TASK 2

model_train <- kknn(V65 ~ ., train=train_df, test=train_df, k=30, kernel="rectangular")
model_test <- kknn(V65 ~ ., train=train_df, test=test_df, k=30, kernel="rectangular")

pred_train <- predict(model_train)
pred_test <- predict(model_test)

confusion_matrix_train <- table(pred_train, train_df$V65)
confusion_matrix_test <- table(pred_test, test_df$V65)

#missclassification rate
MCR <- function(pred_x,x){
  n <- length(pred_x)
  return(1-sum(diag(table(x,pred_x)))/n)
} 

cat("Training Confusion Matrix:\n")
print(table(pred_train, train_df$V65))
print(table(pred_test, test_df$V65))

print(MCR(pred_train, train_df$V65)) #0.04238619
print(MCR(pred_test, test_df$V65)) #0.04502618

#TASK 3

train_8 <- train_df[train_df$V65 == 8,]

train_8_probs <- model_train[["prob"]][train_df$V65 == "8",]


easiest_cases_idx <- sort(train_8_probs[,"8"], decreasing=TRUE, index.return=TRUE)$ix[1:2]
hardest_cases_idx <- sort(train_8_probs[, "8"], decreasing=FALSE, index.return=TRUE)$ix[1:3]

easiest_case_1 <- matrix(as.numeric(train_8[easiest_cases_idx[1],1:64]),nrow=8, ncol=8)
easiest_case_2 <- matrix(as.numeric(train_8[easiest_cases_idx[2],1:64]),nrow=8, ncol=8)
hardest_case_1 <- matrix(as.numeric(train_8[hardest_cases_idx[1],1:64]),nrow=8, ncol=8)
hardest_case_2 <- matrix(as.numeric(train_8[hardest_cases_idx[2],1:64]),nrow=8, ncol=8)
hardest_case_3 <- matrix(as.numeric(train_8[hardest_cases_idx[3],1:64]),nrow=8, ncol=8)

heatmap(t(easiest_case_1), Colv=NA, Rowv=NA)
heatmap(t(easiest_case_2), Colv=NA, Rowv=NA)
heatmap(t(hardest_case_1), Colv=NA, Rowv=NA)
heatmap(t(hardest_case_2), Colv=NA, Rowv=NA)
heatmap(t(hardest_case_3), Colv=NA, Rowv=NA)

#TASK 4

log_prob <- function(p){
  -log(p +1e-15)
}
missclassification_train <- rep(0,30)
missclassification_validation <- rep(0, 30)
cross_entropy <- rep(0,30)

for(x in 1:30){
  #fit both models on data
  temp_model_train <- kknn(V65 ~ ., train=train_df, test=train_df, k=x, kernel="rectangular")
  temp_model_validation <- kknn(V65 ~ ., train=train_df, test=validation_df, k=x, kernel="rectangular")
  
  pred_train <- predict(temp_model_train)
  pred_valid <- predict(temp_model_validation)
  
  missclassification_train[x] <- MCR(pred_train, train_df$V65)
  missclassification_validation[x] <- MCR(pred_valid, validation_df$V65)
  
  #calculate cross-entropy over all classes for given K val
  
  for(y in 0:9){
    prob <- temp_model_validation$prob[which(validation_df$V65==y), y+1]
    prob <- sum(sapply(prob, log_prob))
    cross_entropy[x] <- cross_entropy[x] + prob
  }
}

# Plot misclassification errors
plot(1:30, missclassification_train, type="l", col="blue", 
     xlab="K", ylab="Misclassification Error",
     main="Training vs Validation Misclassification Error",
     ylim=range(c(missclassification_train, missclassification_validation)))
lines(1:30, missclassification_validation, col="red")
legend("topright", legend=c("Training", "Validation"), 
       col=c("blue", "red"), lty=1)

optimal_k <- which.min(missclassification_validation)
cat("Optimal K:", optimal_k, "\n")

# Test error for optimal K
optimal_k_model_test <- kknn(V65 ~ ., train=train_df, test=test_df, k=optimal_k, kernel="rectangular")
pred_test_optimal <- predict(optimal_k_model_test)
test_error_optimal <- MCR(pred_test_optimal, test_df$V65)

cat("Training Error at optimal K:", missclassification_train[optimal_k], "\n")
cat("Validation Error at optimal K:", missclassification_validation[optimal_k], "\n")
cat("Test Error at optimal K:", test_error_optimal, "\n")

#TASK 5 plot cross entropy

optimal_k_ce <- which.min(cross_entropy)
cat("Optimal K (cross-entropy):", optimal_k_ce, "\n")

plot(cross_entropy,col="red", type="b", xlab="K", ylab="Cross-Entropy")