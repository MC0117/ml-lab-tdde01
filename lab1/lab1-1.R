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


#===============================================================================
# LAB: K-NEAREST NEIGHBORS (KNN) FOR DIGIT CLASSIFICATION
#===============================================================================
# DATASET: Optical recognition of handwritten digits (0-9)
# - 64 features: 8x8 pixel intensity values (0-16)
# - 1 target: digit class (0-9)
# - Total observations: ~3800
#
# KEY CONCEPTS:
# 1. KNN is a NON-PARAMETRIC, INSTANCE-BASED learning algorithm
#    - Makes predictions based on K nearest neighbors in feature space
#    - Distance metric: Euclidean distance (default)
#    - No explicit training phase - stores all training data
#
# 2. MODEL COMPLEXITY vs K:
#    - Small K (e.g., K=1): HIGH complexity, flexible decision boundaries
#      → Can overfit (captures noise), low bias, high variance
#    - Large K (e.g., K=30): LOW complexity, smooth decision boundaries  
#      → Can underfit (too simple), high bias, low variance
#    - Trade-off: Bias-Variance tradeoff!
#
# 3. ERROR METRICS:
#    a) MISCLASSIFICATION RATE = 1 - (correct predictions / total predictions)
#       - Simple: only cares if prediction is right or wrong
#       - Ignores prediction confidence
#    
#    b) CROSS-ENTROPY = -Σ log(p_true_class)
#       - Formula: CE = -Σ_{i=1}^{n} log(P(y_i = true_class | x_i))
#       - Considers prediction CONFIDENCE (probability)
#       - Heavily penalizes confident wrong predictions
#       - Better for probabilistic models with multinomial distribution
#       - Example: P(class=8) = 0.9 vs 0.5 → CE prefers confident correct predictions
#
# 4. CRITICAL: Data split order matters!
#    - Training (50%): Fit the model
#    - Validation (25%): Tune hyperparameters (select optimal K)
#    - Test (25%): Final evaluation (estimate generalization error)
#    - NEVER use test data for model selection!
#
# 5. IMPORTANT R QUIRKS:
#    - read.csv with header=TRUE uses first data row as column names → WRONG!
#    - R indexing starts at 1, not 0 (so digit 0 is in column 1)
#    - kknn with train=test=same data gives fitted values (training error ≈ 0)
#===============================================================================

#TASK 1: DATA PREPARATION
#-------------------------------------------------------------------------------
digits <- read.csv("optdigits.csv", header=FALSE)  # CRITICAL: header=FALSE!
digits[,65] <- as.factor(digits[,65])  # Convert target to factor for classification
n <- dim(digits)[1]
set.seed(12345)  # Reproducibility
idx <- sample(1:n)  # Random permutation of indices

# Split: 50% train, 25% test, 25% validation
train_idx <- idx[1:floor(0.5*n)]
test_idx <- idx[(floor(0.5*n) + 1):floor(0.75*n)]
validation_idx <- idx[(floor(0.75*n) + 1):n]
train_df <- digits[train_idx,]
test_df <- digits[test_idx,]
validation_df <- digits[validation_idx,]

#TASK 2: BASELINE KNN MODEL (K=30)
#-------------------------------------------------------------------------------
# Fit KNN with K=30 (relatively large K → simpler model)
model_train <- kknn(V65 ~ ., train=train_df, test=train_df, k=30, kernel="rectangular")
model_test <- kknn(V65 ~ ., train=train_df, test=test_df, k=30, kernel="rectangular")
pred_train <- predict(model_train)
pred_test <- predict(model_test)

# Misclassification Rate (MCR) function
# Formula: MCR = 1 - (# correct predictions / # total predictions)
MCR <- function(pred_x, x){
  n <- length(pred_x)
  return(1 - sum(diag(table(x, pred_x))) / n)
} 

cat("Training Confusion Matrix:\n")
print(table(pred_train, train_df$V65))
print(table(pred_test, test_df$V65))
print(MCR(pred_train, train_df$V65))  # Training error
print(MCR(pred_test, test_df$V65))    # Test error

# INTERPRETATION: Training error < Test error suggests some overfitting,
# but both are low (~4-5%) indicating good model performance

#TASK 3: ANALYZING DIGIT "8" PREDICTIONS
#-------------------------------------------------------------------------------
# Goal: Find easiest and hardest digit "8" cases to classify
# Probability interpretation: P(class=8|x) from KNN voting
train_8 <- train_df[train_df$V65 == 8,]
train_8_probs <- model_train[["prob"]][train_df$V65 == "8",]  # Extract probabilities

# Sort by confidence: High P(8) = easy, Low P(8) = hard/ambiguous
easiest_cases_idx <- sort(train_8_probs[,"8"], decreasing=TRUE, index.return=TRUE)$ix[1:2]
hardest_cases_idx <- sort(train_8_probs[, "8"], decreasing=FALSE, index.return=TRUE)$ix[1:3]

# Reshape 64 features → 8x8 pixel matrix for visualization
easiest_case_1 <- matrix(as.numeric(train_8[easiest_cases_idx[1],1:64]), nrow=8, ncol=8)
easiest_case_2 <- matrix(as.numeric(train_8[easiest_cases_idx[2],1:64]), nrow=8, ncol=8)
hardest_case_1 <- matrix(as.numeric(train_8[hardest_cases_idx[1],1:64]), nrow=8, ncol=8)
hardest_case_2 <- matrix(as.numeric(train_8[hardest_cases_idx[2],1:64]), nrow=8, ncol=8)
hardest_case_3 <- matrix(as.numeric(train_8[hardest_cases_idx[3],1:64]), nrow=8, ncol=8)

# Heatmap visualization (Rowv=NA, Colv=NA disables clustering)
# t() transposes because heatmap plots differently than matrix indexing
heatmap(t(easiest_case_1), Colv=NA, Rowv=NA)
heatmap(t(easiest_case_2), Colv=NA, Rowv=NA)
heatmap(t(hardest_case_1), Colv=NA, Rowv=NA)
heatmap(t(hardest_case_2), Colv=NA, Rowv=NA)
heatmap(t(hardest_case_3), Colv=NA, Rowv=NA)

# EXPECTED: Easiest = clear, well-formed 8s; Hardest = ambiguous/malformed

#TASK 4: HYPERPARAMETER TUNING - FINDING OPTIMAL K
#-------------------------------------------------------------------------------
# GOAL: Test K=1 to K=30 and find optimal K using validation error
# 
# KEY INSIGHT: This demonstrates the bias-variance tradeoff!
# - Small K: High variance (overfits training, sensitive to noise)
# - Large K: High bias (underfits, too smooth)
# - Optimal K: Sweet spot that minimizes validation error

log_prob <- function(p){
  -log(p + 1e-15)  # Add small constant to avoid log(0) = -Inf
}

missclassification_train <- rep(0, 30)
missclassification_validation <- rep(0, 30)
cross_entropy <- rep(0, 30)

for(x in 1:30){
  # Fit models with current K value
  temp_model_train <- kknn(V65 ~ ., train=train_df, test=train_df, k=x, kernel="rectangular")
  temp_model_validation <- kknn(V65 ~ ., train=train_df, test=validation_df, k=x, kernel="rectangular")
  
  pred_train <- predict(temp_model_train)
  pred_valid <- predict(temp_model_validation)
  
  # Compute misclassification rates
  missclassification_train[x] <- MCR(pred_train, train_df$V65)
  missclassification_validation[x] <- MCR(pred_valid, validation_df$V65)
  
  # CROSS-ENTROPY CALCULATION
  # Formula: CE = -Σ_{all classes} Σ_{samples in class} log(P(true_class | x))
  # For each true digit class (0-9), sum log probabilities
  for(y in 0:9){
    # Extract probabilities for samples where true class = y
    # IMPORTANT: R indexing - digit 0 is column 1, so use y+1
    prob <- temp_model_validation$prob[which(validation_df$V65 == y), y+1]
    prob <- sum(sapply(prob, log_prob))  # Sum of -log(p) for this class
    cross_entropy[x] <- cross_entropy[x] + prob  # Accumulate across all classes
  }
}

# PLOT: Training vs Validation Misclassification Error
# EXPECTED PATTERN:
# - Training error: Low or 0 for small K (especially K=1), increases with K
# - Validation error: U-shaped curve (high at K=1, decreases, then increases)
plot(1:30, missclassification_train, type="l", col="blue", 
     xlab="K", ylab="Misclassification Error",
     main="Training vs Validation Misclassification Error",
     ylim=range(c(missclassification_train, missclassification_validation)))
lines(1:30, missclassification_validation, col="red")
legend("topright", legend=c("Training", "Validation"), 
       col=c("blue", "red"), lty=1)

# Find optimal K (minimum validation error)
optimal_k <- which.min(missclassification_validation)
cat("Optimal K:", optimal_k, "\n")

# FINAL EVALUATION: Test error with optimal K
# This estimates how well the model generalizes to unseen data
optimal_k_model_test <- kknn(V65 ~ ., train=train_df, test=test_df, k=optimal_k, kernel="rectangular")
pred_test_optimal <- predict(optimal_k_model_test)
test_error_optimal <- MCR(pred_test_optimal, test_df$V65)

cat("Training Error at optimal K:", missclassification_train[optimal_k], "\n")
cat("Validation Error at optimal K:", missclassification_validation[optimal_k], "\n")
cat("Test Error at optimal K:", test_error_optimal, "\n")

# INTERPRETATION:
# - If test error ≈ validation error: Good! Model generalizes well
# - If test error >> validation error: Possible overfitting to validation set
# - Training error = 0 for KNN is common (points are their own neighbors)

#TASK 5: CROSS-ENTROPY AS ERROR METRIC
#-------------------------------------------------------------------------------
# PLOT: Cross-Entropy vs K
plot(cross_entropy, col="red", type="b", pch=5, xlab="K", ylab="Cross-Entropy")

optimal_k_ce <- which.min(cross_entropy)
cat("Optimal K (cross-entropy):", optimal_k_ce, "\n")

# WHY CROSS-ENTROPY > MISCLASSIFICATION ERROR?
# summary:
# 1. CONFIDENCE MATTERS:
#    - MCR: Treats P(8)=0.51 and P(8)=0.99 the same (both correct)
#    - CE: Rewards confident correct predictions, penalizes uncertain ones
#
# 2. PROBABILISTIC INTERPRETATION:
#    - Assumes multinomial distribution over classes
#    - Maximizing likelihood = Minimizing cross-entropy
#    - Better aligns with probabilistic nature of KNN predictions
#
# 3. DIFFERENTIABLE & SMOOTH:
#    - CE is continuous and smooth (good for optimization)
#    - MCR is step function (0 or 1) - harder to optimize
#
# 4. EARLY WARNING:
#    - CE can detect problems before MCR (model becoming less confident)
#    - Example: Model still predicts correctly but P(true_class) drops 0.9→0.6
#
# FORMULA RECAP:
# MCR = (# misclassified) / (# total)
# CE = -Σ log(P(y_true | x)) = penalizes low probabilities exponentially
#
# EXPECTED: Optimal K from CE might differ slightly from MCR optimal K
#===============================================================================
# KEY TAKEAWAYS:
# 1. Always use validation set for hyperparameter tuning (K selection)
# 2. Test set is ONLY for final evaluation - use once!
# 3. KNN complexity inversely related to K (small K = complex)
# 4. Cross-entropy better captures prediction quality than simple accuracy
# 5. R indexing: digit 0 → column 1 (always add 1 to class labels!)
#===============================================================================