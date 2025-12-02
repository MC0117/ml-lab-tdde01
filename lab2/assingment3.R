communities = read.csv("communities.csv", header = TRUE)

#Question 1

#extract the crimerate
crime_rate <- communities[,101]


library(caret)
# names(communities) gives ViolentCrimesPerPop index 101
scaler = preProcess(communities[,-101], method = c("center", "scale"))
communities_scaled <- predict(scaler, communities[,-101])

#calculate the covariance matrix
cov <- cov(communities_scaled)
pca_results_1 <- eigen(cov) #solving equation C x u = lambda * u
eigen_values <- pca_results_1$values

total_variance <- sum(eigen_values) #100 after scaling

variance_proportion <- eigen_values/total_variance

cumulative_variance <- cumsum(eigen_values/total_variance)

#Amount of components needed for 95% of the variance
components_for_95_percent <- which(cumulative_variance >= 0.95)[1]

#The proportion of PC1
variance_proportion[1]

#The proportion of PC2
variance_proportion[2]

#Question 2 (Oral defence)

#Trace plot of loading scores
pca_results_2 = princomp(communities_scaled)
loading_scores_sorted <- sort(pca_results_2$loadings, decreasing=TRUE)
plot(pca_results_2$loadings[,1], main="Trace Plot of PC1 Loading scores", ylab="Loading Score")




sorted_pc1_loadings <- sort(pca_results_2$loadings[,1], decreasing = TRUE)
barplot(sorted_pc1_loadings, 
        main = "PC1 Feature Contributions (Loadings)",
        xlab = "Features",
        ylab = "PC1 Loading Score",
        las = 2,       # rotates x-axis labels vertically
        cex.names = 0.5)

sorted_pc1_loadings <- sort(abs(pca_results_2$loadings[,1]), decreasing = TRUE)
barplot(sorted_pc1_loadings, 
        main = "PC1 Feature Contributions (Loadings)",
        xlab = "Features",
        ylab = "|PC1 Loading Score|",
        las = 2,       # rotates x-axis labels vertically
        cex.names = 0.5)

#No not many varibles have a notable contribution or dominates PC1, 
#the greater values are around 0.15 and 0.18 in absolute magnitude


#set some values (loading, absolute loading, sort them etc)
pc1_loadings <- pca_results_2$loadings[,1]
abs_pc1_loadings <- abs(pc1_loadings)
sorted_pc1_loadings <- sort(abs_pc1_loadings, decreasing=TRUE)

top_5_contributing_features = sorted_pc1_loadings[1:5]
print(top_5_contributing_features)
#greatest contributing features are: medFamInc, medIncome PctKids2Par pctWInvInc, PctPopUnderPov

# medFamInc (Median Family Income)
# medIncome (Median Income)
# PctKids2Par (Percentage of Kids with Two Parents)
# pctWInvInc (Percentage with Investment Income)
# PctPopUnderPov (Percentage of Population Under Poverty)
# one could easily make the case that these features all are correlated to the crimerate.


library(ggplot2)

scores_df <- data.frame(
  PC1 = pca_results_2$scores[,1],
  PC2 = pca_results_2$scores[,2],
  crime_rate = crime_rate
)

ggplot(scores_df, aes(x=PC1, y=PC2, color=crime_rate))  +
         geom_point(alpha=0.7, size=3) + 
         scale_color_gradient(low="blue", high="red")+
         labs(title="PCA Scores Colored by Violent Crime Rate",
              x = "Principal Component 1 (PC1)",
              y = "Principal Component 2 (PC2)",
              color = "Violent Crimes per Pop") +
         theme_minimal()



