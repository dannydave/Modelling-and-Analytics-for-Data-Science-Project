# Machine Learning------------------------------------------------------------

# Part (a)------------------------------------------------

# Load the data
data <- read.table("earthquake.txt", header = TRUE, sep = " ")

# Generate numerical summaries
summary(data)

# Create scatter plots
library(ggplot2)
# Scatter plot of body vs. surface magnitude
ggplot(data, aes(x = body, y = surface, color = type)) +
  geom_point() +
  labs(title = "Scatter plot of body vs. surface magnitude",
       x = "Body-wave magnitude (mb)",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()

# Boxplots to show the distribution of body and surface magnitudes by type
ggplot(data, aes(x = type, y = body, fill = type)) +
  geom_boxplot() +
  labs(title = "Boxplot of body magnitude by type",
       x = "Type",
       y = "Body-wave magnitude (mb)") +
  theme_minimal()

ggplot(data, aes(x = type, y = surface, fill = type)) +
  geom_boxplot() +
  labs(title = "Boxplot of surface magnitude by type",
       x = "Type",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()


# Part (b)----------------------------------------------------

# Load the necessary libraries
library(caret)
library(randomForest)
library(e1071)

# Assuming data is already loaded and processed
# Prepare the data
set.seed(123)  # for reproducibility
data$type <- as.factor(data$type)

# Split the data into features and target variable
features <- data[, c("body", "surface")]
target <- data$type

# Fit SVM model
svm_model <- svm(type ~ ., data = data, kernel = "radial", probability = TRUE)

# Model evaluation with leave-one-out cross-validation using SVM
svm_loocv <- caret::train(type ~ ., data = data, method = "svmRadial",
                          trControl = trainControl(method = "LOOCV", classProbs = TRUE, summaryFunction = twoClassSummary),
                          metric = "ROC")

# Get support vectors
support_vectors <- data[svm_model$index,]

# Plot the data and highlight the support vectors
ggplot(data, aes(x = body, y = surface, color = type)) +
  geom_point() +
  geom_point(data = support_vectors, aes(x = body, y = surface), shape = 8, size = 3, stroke = 2) +
  labs(title = "SVM Support Vectors",
       x = "Body-wave magnitude (mb)",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()


# Predict on new data (replace 'new_data' with your actual new data frame)
new_data <- data.frame(body = c(5.0), surface = c(4.0))  # Example new data point
predictions <- predict(svm_model, newdata = new_data)
probabilities <- attr(predictions, "probabilities")  # Probability estimates

predictions
probabilities

# Create a grid to cover the feature space
grid <- expand.grid(body = seq(min(data$body), max(data$body), length.out = 100),
                    surface = seq(min(data$surface), max(data$surface), length.out = 100))

# Predict on the grid
grid$predict <- predict(svm_model, newdata = grid)

# Plot the decision boundary
ggplot(data, aes(x = body, y = surface, color = type)) +
  geom_point() +
  geom_tile(data = grid, aes(fill = predict, alpha = 0.5), color = NA) +
  scale_fill_manual(values = c("white", "black")) +
  labs(title = "SVM Decision Boundary",
       x = "Body-wave magnitude (mb)",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()


# Fit logistic regression with a lower threshold for convergence
logit_model <- glm(type ~ body + surface, data = data, family = "binomial", 
                   control = glm.control(maxit = 50, epsilon = 1e-8))

coef_logit <- coef(logit_model)

# Function to calculate decision boundary
boundary_f <- function(body) {
  (-coef_logit["(Intercept)"] - coef_logit["body"] * body) / coef_logit["surface"]
}

# Generate a sequence of body values for plotting the decision boundary
body_seq <- seq(from = min(data$body), to = max(data$body), length.out = 100)

# Calculate corresponding surface values on the decision boundary
surface_seq <- boundary_f(body_seq)

# Create a new data frame for plotting the decision boundary
boundary_data <- data.frame(body = body_seq, surface = surface_seq)

# Plot the decision boundary
ggplot(data, aes(x = body, y = surface, color = type)) +
  geom_point() +
  geom_line(data = boundary_data, aes(x = body, y = surface), color = "black") +
  labs(title = "Logistic Regression Decision Boundary",
       x = "Body-wave magnitude (mb)",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()

# Random Forest
# Tune hyperparameters: number of trees (ntree) and number of variables to possibly split at each node (mtry)
tune_rf <- train(type ~ body + surface, data = data, method = "rf",
                 tuneLength = 5,  # number of different parameter combinations
                 trControl = trainControl(method = "LOOCV"))  # leave-one-out cross-validation


# Best tuned Random Forest model
rf_model <- tune_rf$finalModel

# Random Forest decision boundary is not straightforward to plot because it's a non-linear boundary.
# For visualization purposes, we can plot a contour map of predictions.

# Create a grid to predict over
grid <- with(data, expand.grid(body = seq(min(body), max(body), length.out = 100),
                               surface = seq(min(surface), max(surface), length.out = 100)))

# Predict using the random forest
grid$predict <- predict(rf_model, newdata = grid, type = "class")

# Plot the prediction contour map
ggplot(data, aes(x = body, y = surface, color = type)) +
  geom_point() +
  geom_tile(data = grid, aes(fill = predict, alpha = 0.5), color = NA) +
  scale_fill_manual(values = c("white", "black")) +
  labs(title = "Random Forest Prediction Contour", x = "Body-wave magnitude (mb)", y = "Surface-wave magnitude (Ms)") +
  theme_minimal()

# Model evaluation with leave-one-out cross-validation
# Logistic Regression
logit_loocv <- caret::train(type ~ ., data = data, method = "glm",
                            trControl = trainControl(method = "LOOCV", classProbs = TRUE),
                            family = "binomial")

# Random Forest
rf_loocv <- caret::train(type ~ ., data = data, method = "rf",
                         trControl = trainControl(method = "LOOCV"),
                         tuneGrid = data.frame(mtry = tune_rf$bestTune$mtry))


logit_results <- logit_loocv$results
rf_results <- rf_loocv$results

logit_results
rf_results


# Part (d)----------------------------------------------

# Assuming the 'data' dataframe with 'body' and 'surface' columns is already loaded
data_to_cluster <- data[, c("body", "surface")]

# Standardize the data
data_to_cluster <- scale(data_to_cluster)

set.seed(123)  # for reproducibility
wcss <- sapply(1:10, function(k){
  kmeans(data_to_cluster, centers = k, nstart = 25)$tot.withinss
})

# Plot the elbow method to find optimal number of clusters
plot(1:10, wcss, type = "b", xlab = "Number of clusters", ylab = "Within-cluster sum of squares", main = "Elbow Method")

# Let's suppose the elbow method suggested 2 or 3 clusters are optimal
set.seed(123)
kmeans_result_2 <- kmeans(data_to_cluster, centers = 2, nstart = 25)
kmeans_result_3 <- kmeans(data_to_cluster, centers = 3, nstart = 25)

# Adding the clusters to the data frame
data$cluster_2 <- kmeans_result_2$cluster
data$cluster_3 <- kmeans_result_3$cluster

# Plot the clusters
ggplot(data, aes(x = body, y = surface, color = factor(cluster_2))) +
  geom_point() +
  labs(title = "K-Means Clustering with K=2",
       x = "Body-wave magnitude (mb)",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()

ggplot(data, aes(x = body, y = surface, color = factor(cluster_3))) +
  geom_point() +
  labs(title = "K-Means Clustering with K=3",
       x = "Body-wave magnitude (mb)",
       y = "Surface-wave magnitude (Ms)") +
  theme_minimal()

# Calculate the average silhouette width for 2 and 3 clusters
library(cluster)
sil_width_2 <- silhouette(kmeans_result_2$cluster, dist(data_to_cluster))
mean(sil_width_2[, 3])

sil_width_3 <- silhouette(kmeans_result_3$cluster, dist(data_to_cluster))
mean(sil_width_3[, 3])



# Bayesian Statistics One-way Analysis of Variance----------------------------------------------------------
# First Sub-task
# Bayesian Statistics Part (a)-------------------------------------------------
airline_data <- read.csv('airline.csv')

# Create a boxplot of satisfaction scores for each airline
ggplot(airline_data, aes(x = airline, y = satisfactionscore, fill = airline)) +
  geom_boxplot() +
  labs(title = "Customer Satisfaction Scores by Airline",
       x = "Airline",
       y = "Satisfaction Score") +
  stat_summary(fun = mean, 
               geom = "point", 
               shape = 18, 
               size = 3, 
               color = "darkblue", 
               show.legend = FALSE) +
  stat_summary(fun = mean, 
               geom = "text", 
               aes(label = round(..y.., digits = 2)), 
               color = "darkblue", 
               show.legend = FALSE, 
               vjust = -0.7) +
  theme_minimal()



# Bayesian Statistics Part (c)--------------------------------
# Fit the linear model
model <- lm(satisfactionscore ~ airline, data = airline_data)

# Summarise the model to get estimates
summary(model)

# ANOVA to test the hypothesis
anova_model <- anova(model)
anova_model
summary(anova_model)
coef(anova_model)


#Bayesian Statistics Part (d)-------------------------------------------
# Perform Tukey HSD test
tukey_test <- TukeyHSD(aov(satisfactionscore ~ airline, data = airline_data))
tukey_test


# Bayesian Statistics Two-ways Analysis of Variance---------------------------------------------------------

# Second Sub-task
# Bayesian Statistics Part f-------------------------------------------------------
library(R2jags)
library(coda)
library(ggplot2)
library(ggmcmc)

two_way_anova <- function(){
  # Define the data model
  for(i in 1:I){ # Loop across fields
    for(j in 1:J){ # Loop across treatments
      y_mat[i,j] ~ dnorm(mu[i,j], tau) # Parametrized by the precision tau = 1 / sigma^2
      mu[i,j] <- m + alpha[i] + beta[j]
    }
  }
  
  # Constraints for identifiability
  alpha[1] <- 0 # Corner constraint
  beta[1] <- 0 # Corner constraint
  
  # Priors on unknown parameters
  m ~ dnorm(0.0, 1.0E-4) # Prior on m
  
  for(i in 2:I){
    alpha[i] ~ dnorm(0.0, 1.0E-4) # Prior on non-constrained alphas
  }
  
  for(j in 2:J){
    beta[j] ~ dnorm(0.0, 1.0E-4) # Prior on non-constrained betas
  }
  
  tau ~ dgamma(1.0E-3, 1.0E-3) # Prior on tau
  #
  # Also monitor sigma
  #
  sigma <- 1.0 / sqrt(tau) # Definition of sigma
}

# Data
y <- c(
  208, 216, 220, 226, 209,
  194, 212, 218, 239, 224,
  199, 211, 227, 227, 221
)

#
field <- gl(3, 5, 15) # 3 levels, each repeated 5 times, a total of 15
field

#
technique <- gl(5, 1, 15) # 5 levels, each repeated once, a total of 15
technique
I <- 3 # Number of fields
J <- 5 # Number of techniques

# Express y as a matrix
y_mat <- matrix(y, byrow = TRUE, nrow = I, ncol = J)
y_mat

# Run the Bayesian analysis
data_two_way_anova <- list("y_mat", "I", "J")
#
Bayesian_two_way_anova <- jags(
  data = data_two_way_anova,
  parameters.to.save = c("m", "alpha", "beta", "tau", "sigma"),
  n.iter = 100000,
  n.chains = 3,
  model.file = two_way_anova
)

# Summarize the posterior probability density functions
print(Bayesian_two_way_anova, intervals = c(0.025, 0.5, 0.975))



Bayesian_two_way_anova.mcmc <- as.mcmc(Bayesian_two_way_anova)
Bayesian_two_way_anova.ggs <- ggs(Bayesian_two_way_anova.mcmc)


# Bayesian Statistics Part g----------------------------------------------------------------
# Plot the traceplots and posterior densities

ggs_traceplot(Bayesian_two_way_anova.ggs, family = "m")

ggs_traceplot(Bayesian_two_way_anova.ggs, family="alpha")

ggs_traceplot(Bayesian_two_way_anova.ggs, family="beta")

ggs_density(Bayesian_two_way_anova.ggs, family="m") + xlim(-30, 250)

ggs_density(Bayesian_two_way_anova.ggs, family="alpha") + xlim(-15, 25)

ggs_density(Bayesian_two_way_anova.ggs, family="beta") + xlim(-10, 55)


# Bayesian Statistics Part h---------------------------------------------------------------
# Plot of 95% credible intervals for the parameters

ggs_caterpillar(Bayesian_two_way_anova.ggs, family="m") + xlim(-30, 250)

ggs_caterpillar(Bayesian_two_way_anova.ggs, family="alpha") + xlim(-15, 25)

ggs_caterpillar(Bayesian_two_way_anova.ggs, family="beta") + xlim(-10, 55)


summary(Bayesian_two_way_anova)

plot(as.mcmc(Bayesian_two_way_anova))

m <- lm(y ~ field + technique)
m 

confint(m)

Bayesian_two_way_anova$BUGSoutput$summary[c("m",
                                            "alpha[2]",
                                            "alpha[3]",
                                            "beta[2]",
                                            "beta[3]",
                                            "beta[4]",
                                            "beta[5]"),
                                          c("mean",
                                            "50%",
                                            "2.5%",
                                            "97.5%")]

anova(m)

# Bayesian Statistics part (i)-------------------------------------------------------------------------------------
library(R2jags)
library(coda)
library(ggplot2)
library(ggmcmc)

Bayesian_anova_2 <- function(){
  # Define the data model
  for(i in 1:I){ # Loop across fields
    for(j in 1:J){ # Loop across treatments
      y_mat[i,j] ~ dnorm(mu[i,j], tau) # Parametrized by the precision tau = 1 / sigma^2
      mu[i,j] <- m + alpha[i] + beta[j]
    }
  }
  
  # Constraints for identifiability
  alpha[1] <- 0 # Corner constraint
  beta[1] <- 0 # Corner constraint
  
  # Priors on unknown parameters
  m ~ dnorm(0.0, 1.0E-4) # Prior on m
  
  for(i in 2:I){
    alpha[i] ~ dnorm(0.0, 1.0E-4) # Prior on non-constrained alphas
  }
  
  for(j in 2:J){
    beta[j] ~ dnorm(0.0, 1.0E-4) # Prior on non-constrained betas
  }
  
  # Calculate the differences between beta[4] and the other betas
  delta_beta_4_1 <- beta[4] - beta[1]
  delta_beta_4_2 <- beta[4] - beta[2]
  delta_beta_4_3 <- beta[4] - beta[3]
  delta_beta_4_5 <- beta[4] - beta[5]
  
  tau ~ dgamma(1.0E-3, 1.0E-3) # Prior on tau
  sigma <- 1.0 / sqrt(tau) # Definition of sigma
}

# Data
y <- c(
  208, 216, 220, 226, 209,
  194, 212, 218, 239, 224,
  199, 211, 227, 227, 221
)

#
field <- gl(3, 5, 15) # 3 levels, each repeated 5 times, a total of 15
field

#
technique <- gl(5, 1, 15) # 5 levels, each repeated once, a total of 15
technique
I <- 3 # Number of fields
J <- 5 # Number of techniques

# Express y as a matrix
y_mat <- matrix(y, byrow = TRUE, nrow = I, ncol = J)
y_mat


# Run the Bayesian analysis
data_two_way_anova <- list("y_mat", "I", "J")

# Rest of the code remains the same, but include the new parameters in the parameters.to.save argument
Bayesian_anova_inference_2 <- jags(
  data = data_two_way_anova,
  parameters.to.save = c("m", "alpha", "beta", "delta_beta_4_1", "delta_beta_4_2", "delta_beta_4_3", "delta_beta_4_5", "tau", "sigma"),
  n.iter = 100000,
  n.chains = 3,
  model.file = Bayesian_anova_2
)

# Summarize the posterior probability density functions
print(Bayesian_anova_inference_2, intervals = c(0.025, 0.5, 0.975))

Bayesian_anova_inference_2.mcmc <- as.mcmc(Bayesian_anova_inference_2)
Bayesian_anova_inference_2.ggs <- ggs(Bayesian_anova_inference_2.mcmc)


# Plot the traceplot for the new parameter 
ggs_traceplot(Bayesian_anova_inference_2.ggs, family = "delta_beta_4_1")
ggs_traceplot(Bayesian_anova_inference_2.ggs, family = "delta_beta_4_2")
ggs_traceplot(Bayesian_anova_inference_2.ggs, family = "delta_beta_4_3")
ggs_traceplot(Bayesian_anova_inference_2.ggs, family = "delta_beta_4_5")


# plot the posterior densities for the new parameters
ggs_density(Bayesian_anova_inference_2.ggs, family="delta_beta_4_1")
ggs_density(Bayesian_anova_inference_2.ggs, family="delta_beta_4_2")
ggs_density(Bayesian_anova_inference_2.ggs, family="delta_beta_4_3")
ggs_density(Bayesian_anova_inference_2.ggs, family="delta_beta_4_5")

# Plot of 95% credible intervals for the parameters

ggs_caterpillar(Bayesian_anova_inference_2.ggs, family="delta_beta_4_1")

ggs_caterpillar(Bayesian_anova_inference_2.ggs, family="delta_beta_4_2")

ggs_caterpillar(Bayesian_anova_inference_2.ggs, family="delta_beta_4_3")

ggs_caterpillar(Bayesian_anova_inference_2.ggs, family="delta_beta_4_5")


summary(Bayesian_anova_inference_2)

plot(as.mcmc(Bayesian_anova_inference_2))



# Simpler Bayesian model------------------------------------------------
# Third Sub-task

library(R2jags)
library(coda)
library(ggplot2)
library(ggmcmc)

# Define the simpler Bayesian model
model_string <- function(){
  for(i in 1:I){ # Loop across rows (Machines)
    for(j in 1:J){ # Loop across columns (Operators)
      # Note that y_mat is a matrix
      y_mat[i,j] ~ dnorm(mu[i,j], tau) # Parametrized by the precision tau = 1 / sigmaˆ2
      mu[i,j] <- m + beta[j]
    }
  }
  #
  # Constraints for identifiability
  #
  beta[1] <- 0 # Corner constraint
  #
  # Priors on unknown parameters
  #
  m ~ dnorm(0.0, 1.0E-4) # Prior on m
  #
  for(j in 2:J){
    beta[j] ~ dnorm(0.0, 1.0E-4) # Prior on non-constrained betas
  }
  tau ~ dgamma(1.0E-3, 1.0E-3) # Prior on tau
  #
  # Also monitor sigma
  #
  sigma <- 1.0 / sqrt(tau) # Definition of sigma
}



# Data
y <- c(
  208, 216, 220, 226, 209,
  194, 212, 218, 239, 224,
  199, 211, 227, 227, 221
)

#
field <- gl(3, 5, 15) # 3 levels, each repeated 5 times, a total of 15
field

#
technique <- gl(5, 1, 15) # 5 levels, each repeated once, a total of 15
technique
I <- 3 # Number of fields
J <- 5 # Number of techniques

# Express y as a matrix
y_mat <- matrix(y, byrow = TRUE, nrow = I, ncol = J)
y_mat

# Run the Bayesian analysis
data_simpler_bayesian <- list("y_mat", "I", "J")
#
Simpler_Bayesian_anova <- jags(
  data = data_simpler_bayesian,
  parameters.to.save = c("m", "beta", "tau", "sigma"),
  n.iter = 100000,
  n.chains = 3,
  model.file = model_string
)

# Summarize the posterior probability density functions
print(Simpler_Bayesian_anova, intervals = c(0.025, 0.5, 0.975))


Simpler_Bayesian_anova.mcmc <- as.mcmc(Simpler_Bayesian_anova)
Simpler_Bayesian_anova.ggs <- ggs(Simpler_Bayesian_anova.mcmc)

# Bayesian Statistics part (k)-------------------------------------------------------------------------------------

# Plot the traceplots and posterior densities

ggs_traceplot(Simpler_Bayesian_anova.ggs, family = "m")

ggs_traceplot(Simpler_Bayesian_anova.ggs, family="beta")

ggs_density(Simpler_Bayesian_anova.ggs, family="m") + xlim(-30, 250)

ggs_density(Simpler_Bayesian_anova.ggs, family="beta") + xlim(-10, 55)

# Plot of 95% credible intervals for the parameters

ggs_caterpillar(Simpler_Bayesian_anova.ggs, family="m") + xlim(-30, 250)

ggs_caterpillar(Simpler_Bayesian_anova.ggs, family="beta") + xlim(-10, 55)

summary(Simpler_Bayesian_anova)

plot(as.mcmc(Simpler_Bayesian_anova))

