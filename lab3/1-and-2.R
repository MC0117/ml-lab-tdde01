
library(pracma)
library(neuralnet)

#TASK 1

set.seed(1234567890)

Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin=sin(Var))
tr <- mydata[1:25,] # Training
te <- mydata[26:500,] # Test

winit <- runif(10, min=-1,max=1)
nn_orig <- neuralnet(Sin ~ ., data=tr, hidden=10, startweights = winit)



plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1], predict(nn_orig, te), col="red", cex=1)

#results are good. as graph shows

#TASK 2 - change with custom activation function

#linear activation function
linear_activation <- function(x){
  x
}

nn_linear <- neuralnet(Sin ~ ., data=tr, hidden=10, startweights = winit, act.fct = linear_activation)

plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1], predict(nn_linear, te), col="red", cex=1) #really bad, straight line

# activation function: ReLU
ReLU_activation <- function(x){
  ifelse(x > 0, x, 0)
}

nn_relu <- neuralnet(Sin ~ ., data=tr, hidden=10, startweights = winit, act.fct = ReLU_activation)

plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1], predict(nn_relu, te), col="red", cex=1) #slightly better but still really bad

# activation function: softplus
softplus_activation <- function(x){
  log(1 + exp(x))
}

nn_softplus <- neuralnet(Sin ~ ., data=tr, hidden=10, startweights = winit, act.fct = softplus_activation, linear.output = TRUE)

plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1], predict(nn_softplus, te), col="red", cex=1) #quite good

# TASK 3 - test extrapolation capability with dynamic scaling
Var <- runif(500, 0, 50)
newdata <- data.frame(Var, Sin=sin(Var))

predictions <- predict(nn_orig, newdata)

# Check the ranges
cat("Prediction range:", range(predictions), "\n")
cat("True sin range:", range(newdata$Sin), "\n")

# Determine y-axis limits to include all data
y_min <- min(newdata$Sin, predictions)
y_max <- max(newdata$Sin, predictions)

# Plot with automatic y-axis scaling
plot(tr, cex=2, xlab="Var", ylab="Sin", xlim=c(0, 50), 
     ylim=c(y_min, y_max),
     main="Neural Network Extrapolation Failure")
points(newdata, col="blue", cex=0.5)  # True sine values in [0,50]
points(newdata[,1], predictions, col="red", cex=0.5)  # Predictions

# Add a vertical line to show training boundary
abline(v=10, lty=2, col="gray", lwd=2)
text(10, y_max*0.9, "Trainingboundary", pos=4, col="gray")

legend("bottomright", c("train data [0,10]", "true sin [0,50]", "predictions"),
       col=c("black", "blue", "red"), pch=1, cex=0.8)
#TASK 4

nn_orig$weights

# See what happens to one neuron
Var_values <- c(0, 5, 10, 20, 50)
for(v in Var_values) {
  activation <- 1 / (1 + exp(-(11.96 - 1.80 * v)))
  cat("Var =", v, "→ activation =", round(activation, 6), "\n")
}

# TASK 5 predict x from sin(x) (inverse problem)
set.seed(1234567890)
Var <- runif(500, 0, 10)
mydata <- data.frame(Sin=sin(Var), Var=Var)  # Sin is input, Var is output!

# Train on ALL 500 points
winit <- runif(10, -1, 1)

# Train the network - may need threshold to avoid convergence issues
nn_inverse <- neuralnet(Var ~ Sin, data=mydata, hidden=10, 
                        startweights=winit, threshold=0.1)

# Predict
predictions <- predict(nn_inverse, mydata)

# Plot results
plot(mydata$Sin, mydata$Var, col="blue", cex=0.8,
     xlab="sin(x)", ylab="x",
     main="Predicting x from sin(x) - The Inverse Problem")
points(mydata$Sin, predictions, col="red", cex=0.8)
legend("topleft", c("True x", "Predicted x"),
       col=c("blue", "red"), pch=1, cex=0.8)

# Calculate error
mse <- mean((mydata$Var - predictions)^2)
cat("Mean Squared Error:", mse, "\n")

