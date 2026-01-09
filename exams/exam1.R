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

vars_to_remove <- c("diagnosis")
data_clean <- raw_data[, !(names(raw_data) %in% vars_to_remove)]

id_train <- sample(1:n, floor(n * 0.50))
train_data <- data_clean[id_train, ]
test_data <- data_clean[-id_train, ]














