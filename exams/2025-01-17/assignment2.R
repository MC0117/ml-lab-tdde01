#Neural Network

library(neuralnet)
set.seed(1234567890)
Var <- runif(50, 0, 10)
tr <- data.frame(Var, Sin=sin(Var))
tr1 <- tr[1:25,] # Fold 1
tr2 <- tr[26:50,] # Fold 2

winit <- runif(10, -1, 1)
model.1 <- neuralnet(Sin ~ Var, data=tr1, hidden=10, startweights = winit, threshold = 0.001)
model.2 <- neuralnet(Sin ~ Var, data=tr2, hidden=10, startweights = winit, threshold = 0.001)

pred.1 <- predict(model.1, newdata=tr2)
pred.2 <- predict(model.2, newdata=tr1)

mse_model.1 <- mean((pred.1 - tr2$Sin)^2)
mse_model.2 <- mean((pred.2 - tr1$Sin)^2)

mse_model.1
mse_model.2

df_1 <- data.frame(
  Var = tr2$Var,
  Sin = pred.1
)


df_2 <- data.frame(
  Var = tr1$Var,
  Sin = pred.2
)

plot(tr, col='black')
points(df_2, col='blue')
points(df_1, col="red")
