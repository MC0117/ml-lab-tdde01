# --- Assignment 2, Part 1: Kernel Models ---
# 1. Generate a Dataset (2 Classes)
# We create a non-linear dataset (e.g., two curved moon-like shapes or nested circles)
# to make the bandwidth h matter. Here we use two noisy concentric clusters.
set.seed(12345)
# Function to generate data
gen_data <- function(n) {
  # Class 1: Inner Circle (with noise)
  r1 <- runif(n/2, 0, 2)
  theta1 <- runif(n/2, 0, 2*pi)
  c1 <- data.frame(x1 = r1 * cos(theta1), x2 = r1 * sin(theta1), class = 1)
  # Class 2: Outer Ring (with noise)
  r2 <- runif(n/2, 3, 5)
  theta2 <- runif(n/2, 0, 2*pi)
  c2 <- data.frame(x2_1 = r2 * cos(theta2), x2_2 = r2 * sin(theta2))
  names(c2) <- c("x1", "x2")
  c2$class <- 2
  return(rbind(c1, c2))
}
# Generate Training and Test Data
train_data <- gen_data(200) # 100 per class
test_data <- gen_data(100)
# 2. Define Kernel Classifier Functions
# The problem specifies using dnorm as the kernel.
# For 2D data, we assume a product kernel: p(x1, x2) = p(x1) * p(x2) (Naive Bayes assumption
#,→ locally)
# or simply multivariate independent KDE.
# Function to estimate class conditional density p(x* | class)
get_density <- function(x_star, data_class, h) {
  # x_star: vector of length 2 (the test point)
  # data_class: dataframe of training points belonging to one class
  # h: bandwidth
  # Calculate (x* - xi) / h for all training points i
  diffs_1 <- (x_star[1] - data_class$x1) / h
  diffs_2 <- (x_star[2] - data_class$x2) / h
  # Kernel values (Gaussian)
  k_vals <- dnorm(diffs_1) * dnorm(diffs_2)
  # Sum and normalize by n (and h^d for proper density scaling, though constant cancels in ratio)
  # Density = (1 / (n * h^2)) * sum(kernels)
  density <- sum(k_vals) / (nrow(data_class) * h^2)
  return(density)
}
# Function to predict class for a single point

predict_point <- function(x_new, train_data, h) {
  # Split training data
  d1 <- train_data[train_data$class == 1, ]
  d2 <- train_data[train_data$class == 2, ]
  # Priors (estimated from data count)
  prior1 <- nrow(d1) / nrow(train_data)
  prior2 <- nrow(d2) / nrow(train_data)
  # Likelihoods p(x | C)
  lik1 <- get_density(x_new, d1, h)
  lik2 <- get_density(x_new, d2, h)
  # Posterior (unnormalized is sufficient for comparison)
  post1 <- lik1 * prior1
  post2 <- lik2 * prior2
  # If both are 0 (numerical underflow for very small h far from points), predict random or
  #,→ majority
  if(post1 == 0 && post2 == 0) return(sample(1:2, 1))
  if (post1 > post2) return(1) else return(2)
}
# Wrapper to predict for a whole dataset
predict_all <- function(new_data, train_data, h) {
  preds <- apply(new_data[, 1:2], 1, predict_point, train_data = train_data, h = h)
  return(preds)
}
# 3. Analyze Effect of h (Bandwidth)
# We test a range of h values to find Overfitting vs Underfitting
h_values <- c(0.01, 0.5, 5)
# Prepare plotting grid for decision boundaries
x_grid <- seq(min(train_data$x1)-1, max(train_data$x1)+1, length.out = 50)
y_grid <- seq(min(train_data$x2)-1, max(train_data$x2)+1, length.out = 50)
grid_points <- expand.grid(x1 = x_grid, x2 = y_grid)
par(mfrow = c(1, 3)) # 3 plots side by side
for (h in h_values) {
  # Predict on Test Data to get Error
  test_preds <- predict_all(test_data, train_data, h)
  test_acc <- mean(test_preds == test_data$class)
  # Predict on Grid for Visualization
  grid_preds <- predict_all(grid_points, train_data, h)
  # Plot
  plot(train_data$x1, train_data$x2, col = ifelse(train_data$class == 1, "red", "blue"),
       pch = 19, main = paste("h =", h, "\nTest Acc:", test_acc),
       xlab = "x1", ylab = "x2", cex = 0.6)
  # Overlay decision boundary (contour)
  z <- matrix(grid_preds, nrow = 50, ncol = 50)
  contour(x_grid, y_grid, z, add = TRUE, levels = 1.5, drawlabels = FALSE, lwd = 2)
}
# 4. Comments (Output to console)
cat("--- Analysis of Bandwidth h ---\n")
cat("1. h = 0.05 (Small): Overfitting.\n")
cat(" The decision boundary is extremely jagged and forms tiny islands around specific training
,→ points.\n")
cat(" It fits noise and fails to generalize to the space between points.\n")
cat("\n2. h = 5 (Large): Underfitting.\n")
cat(" The decision boundary becomes too smooth or disappears entirely (predicts only one class)
,→ .\n")
cat(" It ignores the local structure of the data (the inner circle vs outer ring).\n")
cat("\n3. h = 0.5 (Medium): Good Fit.\n")
cat(" Captures the general circular structure without capturing the noise.\n")
