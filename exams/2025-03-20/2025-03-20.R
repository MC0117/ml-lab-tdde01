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


#############NEXT PART########################

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



