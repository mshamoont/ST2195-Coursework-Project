# Question 1 b
# Load necessary libraries
library(dplyr)
library(tidyr)

library(ggplot2)
library(reshape2)

library(DBI)
library(RSQLite)
#1.b




# Define the target probability density function f(x)
f <- function(x) {
  return(0.5 * exp(-abs(x)))
}

# Metropolis-Hastings algorithm for a single chain
metropolis_hastings_chain <- function(x0, N, s, burn_in = 500) {
  x_values <- numeric(N + burn_in)  # Create space for burn-in + N samples
  x_values[1] <- x0
  
  for (i in 2:(N + burn_in)) {
    # Simulate x* from the normal distribution N(x[i-1], s^2)
    x_star <- rnorm(1, mean = x_values[i - 1], sd = s)
    
    # Compute log acceptance ratio log_r = log(f(x*)) - log(f(x[i-1]))
    log_r <- log(f(x_star)) - log(f(x_values[i - 1]))
    
    # Generate u from Uniform(0, 1)
    u <- runif(1)
    
    # Accept or reject the new value based on the log ratio
    if (log(u) < log_r) {
      x_values[i] <- x_star  # Accept the new value
    } else {
      x_values[i] <- x_values[i - 1]  # Reject the new value
    }
  }
  
  # Return the chain after the burn-in period
  return(x_values[(burn_in + 1):(N + burn_in)])
}

# Function to compute the R_hat value
compute_r_hat <- function(chains) {
  J <- nrow(chains)  # Number of chains
  N <- ncol(chains)  # Number of samples in each chain
  
  # Compute Mj (mean of each chain) and Vj (variance of each chain)
  Mj <- apply(chains, 1, mean)
  Vj <- apply(chains, 1, var)
  
  # Overall mean M
  M <- mean(Mj)
  
  # Compute within-sample variance W
  W <- mean(Vj)
  
  # Compute between-sample variance B
  B <- mean((Mj - M)^2)
  
  # Compute R_hat
  R_hat <- sqrt((B + W) / W)
  return(R_hat)
}

# Generate multiple chains and compute R_hat over a grid of s values
N <- 2000  # Number of steps in each chain
J <- 4  # Number of chains
burn_in <- 500  # Number of samples to discard (burn-in)
initial_values <- c(0, 1, -1, 2)  # Different initial values for each chain
s_values <- seq(0.01, 1, length.out = 100)  # Grid of s values
r_hat_values <- c()

# Set a cap for R_hat values
r_hat_cap <- 2  # Cap for R_hat values

for (s in s_values) {
  chains <- matrix(nrow = J, ncol = N)
  
  for (j in 1:J) {
    chain <- metropolis_hastings_chain(initial_values[j], N, s, burn_in = burn_in)
    chains[j, ] <- chain
  }
  
  # Compute R_hat
  r_hat <- compute_r_hat(chains)
  
  # Cap the R_hat value for visualization
  if (r_hat > r_hat_cap) {
    r_hat <- r_hat_cap
  }
  
  r_hat_values <- c(r_hat_values, r_hat)
}

# Plot R_hat as a function of s
r_hat_df <- data.frame(s_values = s_values, r_hat_values = r_hat_values)

ggplot(r_hat_df, aes(x = s_values, y = r_hat_values)) +
  geom_line() +
  geom_hline(yintercept = 1.05, linetype = "dashed", color = "red") +
  labs(
    title = expression(hat(R) ~ "values for different step sizes s (capped)"),
    x = "Step size (s)",
    y = expression(hat(R) ~ "value")
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

