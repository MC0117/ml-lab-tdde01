# Perform principal component analysis (PCA) on the scaled original data,
# excluding the pH variable.
# Determine how much variation is explained by the first two principal components.
# Identify which features contribute most to the first principal component.
# Assuming only the first two principal components are retained,
# report an equation showing how the unscaled Cond variable
# can be approximated using the first two principal components.

library(caret)
library(dplyr)
data <- read.csv("lakesurvey.csv")

set.seed(12345)
scaler <- preProcess(data)
data_scaled <- predict(scaler, data) #this is weird probably should remove ph first but the text is unclear

data.pH <- data_scaled %>% select(pH)
data_scaled <- data_scaled %>% select(-pH)

cov_matrix <- cov(data_scaled)
pca_eigen <- eigen(cov_matrix)  #eigen vectors and values of principal components
eigen_values <- pca_eigen$values #the variance of each Principal component (sorted)

var_explained <- eigen_values/sum(eigen_values) # how much of total variance each PC contains
cum_var_explained <- cumsum(var_explained) #as we sum up the variance what is the sum ta each step, last will be 100%


#So4 contributes 0.76302820 Cl contributes Cond contributes -0.40659310
#94,76734% of variance of two first PC
#num_components_95 <- which(cum_var_explained >= 0.95)[1] #take first collection of PC:s where sum of PCA variance exeeds 95%

#okay pivot, cannot extract PC1 here, at least I dont know how

data_pca <- data%>%select(-pH)
pca_model <- prcomp(data_pca, scale.=TRUE)

summary(pca_model)


PC1 <- pca_model$rotation[,1]
loadings_sorted_pc1 <- sort(PC1, decreasing =TRUE) #most contributing is Cond Alk Ca SO4 Mg Na Cl





