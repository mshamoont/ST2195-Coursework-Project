#Question 1 a:
#Importing Liabraries
install.packages("ggplot2")
install.packages("RSQLite")
install.packages("dplyr")
install.packages("DBI")
install.packages("reshape2")

library(dplyr)
library(tidyr)

library(ggplot2)
library(reshape2)

library(DBI)
library(RSQLite)

# Define the target probability density function f(x)
f <- function(x) {
  return(0.5 * exp(-abs(x)))
}

# Metropolis-Hastings algorithm for random walk Metropolis
metropolis_hastings <- function(N, s, x0) {
  x_values <- numeric(N)  # Initialize array to store the samples
  x_values[1] <- x0  # Set the initial value
  
  for (i in 2:N) {
    # Simulate x* from the normal distribution N(x[i-1], s^2)
    x_star <- rnorm(1, mean = x_values[i - 1], sd = s)
    
    # Compute the ratio r(x*, xi-1) = f(x*) / f(xi-1)
    log_r <- log(f(x_star)) - log(f(x_values[i - 1]))  # Use log to avoid numerical issues
    
    # Generate a random number u from Uniform(0, 1)
    u <- runif(1)
    
    # Accept or reject the new value based on the ratio
    if (log(u) < log_r) {
      x_values[i] <- x_star  # Accept the new value
    } else {
      x_values[i] <- x_values[i - 1]  # Reject the new value, stay at current value
    }
  }
  
  return(x_values)
}

# Parameters
N <- 10000  # Number of iterations
s <- 1.0    # Standard deviation for proposal distribution
x0 <- 0     # Initial value

# Generate samples using Metropolis-Hastings algorithm
samples <- metropolis_hastings(N, s, x0)

# The generated samples are stored in the 'samples' variable

#Solution 1 

# Step 2: Random walk Metropolis-Hastings
metropolis_hastings <- function(N, s, x0) {
  x_values <- numeric(N + 1)  # Initialize array to store the samples
  x_values[1] <- x0  # Set the initial value
  
  for (i in 2:(N + 1)) {
    # Simulate x* from the normal distribution N(x[i-1], s^2)
    x_star <- rnorm(1, mean = x_values[i - 1], sd = s)
    
    # Compute the log ratio to avoid numerical errors
    log_r <- log(f(x_star)) - log(f(x_values[i - 1]))  # log r(x*, x[i-1])
    
    # Generate u from Uniform(0, 1)
    u <- runif(1)
    
    # Accept or reject the new value based on the log ratio
    if (log(u) < log_r) {
      x_values[i] <- x_star  # Accept the new value
    } else {
      x_values[i] <- x_values[i - 1]  # Reject the new value
    }
  }
  
  return(x_values)
}

# Define the target probability density function f(x)
f <- function(x) {
  return(0.5 * exp(-abs(x)))
}

# Parameters
N <- 10000  # Number of iterations
s <- 1.0    # Standard deviation for proposal distribution
x0 <- 0     # Initial value

# Generate samples using Metropolis-Hastings algorithm
x_values <- metropolis_hastings(N, s, x0)

# Calculate the sample mean and standard deviation
sample_mean <- mean(x_values)
sample_std <- sd(x_values)

# Print the sample mean and standard deviation
cat("Sample Mean:", sample_mean, "\n")
cat("Sample Standard Deviation:", sample_std, "\n")

# Plotting
library(ggplot2)

# Create histogram and kernel density estimate
x_values_trimmed <- x_values[-1]  # Remove the first value (initial state)

# Create a histogram and kernel density plot
ggplot(data.frame(x_values_trimmed), aes(x = x_values_trimmed)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "lightgreen", alpha = 0.6) +
  geom_density(color = "blue", size = 1) +
  stat_function(fun = f, color = "red", size = 1, linetype = "dashed") +
  labs(
    title = "Metropolis-Hastings Sampling",
    x = "x",
    y = "Density"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_vline(xintercept = sample_mean, color = "black", linetype = "dashed") +
  geom_vline(xintercept = sample_mean + sample_std, color = "purple", linetype = "dotted") +
  geom_vline(xintercept = sample_mean - sample_std, color = "purple", linetype = "dotted")

