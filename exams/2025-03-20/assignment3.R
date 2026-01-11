library(neuralnet)
set.seed(1234567890)
x1 <- runif(1000, 0, 2)
x2 <- runif(1000, 0, 2)
tr <- data.frame(x1,x2, y=x1 + x2)
winit <- runif(9, -1, 1)
nn<-neuralnet(formula = y ~ x1 + x2, data = tr, hidden = c(1), act.fct = "tanh")
plot(nn)

tanh_pred <- predict(nn, newdata = tr)

nn$result.matrix # this gives additional info
# Explanation:
# The weight of x1 and x2 to hidden layer is of same magnitude, since x2 is subracted
# from x1 x2 will be negative and x1 weight will be positive.
# Bias is very small, which is reasonable since it should not exist
# the two weights are therefore very close in absolute magnitude (~0.1355...)
# Tanh is approximatly linear for small values so the weight that are learned are also small
# the output of the activation function is then multiplied to a compensating factor 7,45...
# and adds a almost zero bias to retain approximately linear properties

MSE <- function(x, x_pred){
  n <- nrow(x)
  return(sum((x - x_pred)^2)/n)
}

#MY OWN EXPERIMENT
ReLU <- function(z){
  ifelse(z > 0,z,0)
} 

winit <- runif(9, -1, 1)
nn_2<-neuralnet(formula = y ~ x1 + x2, data = tr, hidden = c(1,1), act.fct = ReLU)
plot(nn_2)

ReLU_pred <- predict(nn_2, tr)

MSE(tanh_pred, tr)
MSE(ReLU_pred, tr)

set.seed(54321)
x1 <- runif(1000, -1, 1)
x2 <- runif(1000, -1, 1)
ts <- data.frame(x1,x2, y=x1 - x2)

ReLU_pred <- predict(nn_2, newdata=ts)
tanh_pred <- predict(nn, newdata = ts)

MSE(tanh_pred, ts$y)
MSE(ReLU_pred, ts$y)

nn_2$result.matrix

set.seed(1234567)
x1_new <- runif(1000, -0.8, 0.8)
x2_new <- runif(1000, -0.8, 0.8)
ts_new <- data.frame(x1=x1_new, x2=x2_new, y=x1_new - x2_new)

tanh_pred_new <- predict(nn, newdata = ts_new)
ReLU_pred_new <- predict(nn_2, newdata = ts_new)

MSE(tanh_pred_new, ts_new$y)
MSE(ReLU_pred_new, ts_new$y)