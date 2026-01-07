#ASSIGNMENT 3 logistic regression

#TASK 1
data <- read.csv("pima-indians-diabetes.csv", header=FALSE)

#set.seed(12345)
#n = dim(data)[1]
#idx <- sample(1:n, floor(1.0*n))

#train_df <- data[1:n,]

train_df <- data

diabetes <- train_df$V9
pg_con <- train_df$V2
age <- train_df$V8

plot(age, pg_con, col=ifelse(diabetes == 1, "red", "blue"))
#not easy to classify because it is a multitude of factors that causes it therefore
#finding one threshold that gives classifies is really hard

#TASK 2
r <- 0.5
#train the model
model <- glm(diabetes ~ age + pg_con, data = train_df, family = "binomial")

#some stats
summary(model)

#run predictions on dataset
pred_diabetes <- predict(model, train_df, type = "response") #response to convert from log odds

#plot with predictions
plot(age, pg_con, col=ifelse(pred_diabetes > r, "red", "blue"))


#two variants of calculating MCR
MCE <- function(pred_x, x, r){
  1-(sum((pred_x>r)==x)/length(x))
}

MCE_cm <- function(cm,pred, r){
  n <- length(pred)
  1 - sum(diag(cm))/n
}

MCE(pred_diabetes, diabetes, r) #0.2630208

confusion_matrix <- table(pred_diabetes>r, diabetes)

# Logistic regression model:
# P(Diabetes = 1 | Age, Plasma glucose) =
# 1 / (1 + exp(-( -5.9124 + 0.0248 * Age + 0.0356 * PlasmaGlucose )))

# The training misclassification error is approximately 0.26, meaning about 26% of
# observations are incorrectly classified on the training set.

# The scatter plot of Age vs Plasma glucose concentration colored by predicted class
# shows only a rough separation between diabetic and non-diabetic cases.
# This indicates that while plasma glucose and age are informative,
# they are not sufficient alone to achieve high classification accuracy.

#TASK 3

# Scatter plot colored by predicted class
plot(age, pg_con, col=ifelse(pred_diabetes > r, "red", "blue"),
     pch=19, xlab="Age", ylab="Plasma glucose", main="Predicted Diabetes with decision boundary")

# Add logistic regression decision boundary
abline(a = -coef(model)[1]/coef(model)[3], # intercept: -β0/β2
       b = -coef(model)[2]/coef(model)[3], # slope: -β1/β2
       col="black", lwd=2)

legend("bottomright", legend=c("Predicted 0","Predicted 1","Decision boundary"),
       col=c("blue","red","black"), pch=c(19,19,NA), lty=c(NA,NA,1))

#TASK 4

r <- 0.2

# Scatter plot colored by predicted class
plot(age, pg_con, col=ifelse(pred_diabetes > r, "red", "blue"),
     pch=19, xlab="Age", ylab="Plasma glucose", main="Predicted Diabetes with decision boundary")

# Add logistic regression decision boundary
abline(a = -coef(model)[1]/coef(model)[3], # intercept: -β0/β2
       b = -coef(model)[2]/coef(model)[3], # slope: -β1/β2
       col="black", lwd=2)

legend("bottomright", legend=c("Predicted 0","Predicted 1","Decision boundary"),
       col=c("blue","red","black"), pch=c(19,19,NA), lty=c(NA,NA,1))


#with threshold 0.8

r <- 0.8

# Scatter plot colored by predicted class
plot(age, pg_con, col=ifelse(pred_diabetes > r, "red", "blue"),
     pch=19, xlab="Age", ylab="Plasma glucose", main="Predicted Diabetes with decision boundary")

# Add logistic regression decision boundary
abline(a = -coef(model)[1]/coef(model)[3], # intercept: -β0/β2
       b = -coef(model)[2]/coef(model)[3], # slope: -β1/β2
       col="black", lwd=2)

legend("bottomright", legend=c("Predicted 0","Predicted 1","Decision boundary"),
       col=c("blue","red","black"), pch=c(19,19,NA), lty=c(NA,NA,1))

#MOTIVATION: By increasing r we are increasing the threshold of which probability the model needs to predict 
# in order to classify a case as Positive Class.

# TASK 5

r <- 0.5

# Perform feature engineering on existing features.
train_df$z1 <- age^4
train_df$z2 <- age^3*pg_con^1
train_df$z3 <- age^2*pg_con^2
train_df$z4 <- age^1*pg_con^3
train_df$z5 <- pg_con^4

model_expanded <- glm(diabetes ~ age + pg_con + z1 + z2 + z3 + z4 + z5, data=train_df, family="binomial")

summary(model_expanded)

pred_diabetes <- predict(model_expanded, train_df, type="response")

expanded_confusion_matrix <- table(ifelse(pred_diabetes > r, 1, 0), diabetes)

MCE_cm(expanded_confusion_matrix, pred_diabetes, r) #improved slightly approx 24%
