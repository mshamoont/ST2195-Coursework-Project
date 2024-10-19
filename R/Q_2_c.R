# Question 2 (c): For each year, fit a logistic regression model for the probability of diverted US flights 
#using as many features as possible from attributes of the departure date, the sched- uled departure and arrival times, the coordinates and distance between departure and planned arrival airports, and the carrier. Visualize the coefficients across years.


# Load necessary libraries
library(DBI)
library(RSQLite)
library(dplyr)
library(ggplot2)
library(caret)
library(ROSE)
library(ROCR)

# Database connection (update with your actual database file path)
conn <- dbConnect(RSQLite::SQLite(), dbname = "/Users/muhammadshamoontariq/Desktop/LSE_Prog practice assignments/dataverse_files/flights_data.db")

# List of years from 1996 to 2005 for the analysis
years_of_analysis <- 1996:2005

# Initialize an empty list to hold data for all years
all_years_data <- list()

# Querying data in a loop
for (year in years_of_analysis) {
  print(paste("Querying data for the year:", year))
  
  query <- paste0('
    SELECT 
      f.Month, 
      f.DayofMonth,        
      f.Diverted, 
      f.CRSArrTime AS ScheduledArrival, 
      f.Distance,
      c.Description AS CarrierName,
      a1.long AS Origin_Longitude,
      a2.long AS Dest_Longitude
    FROM 
      "', year, '" f
    LEFT JOIN 
      "carriers" c ON f.UniqueCarrier = c.Code
    LEFT JOIN 
      "airports" a1 ON f.Origin = a1.iata 
    LEFT JOIN 
      "airports" a2 ON f.Dest = a2.iata   
    WHERE 
      f.Diverted IS NOT NULL;')
  
  # Querying data for the year
  df_year <- dbGetQuery(conn, query)
  df_year$Year <- year  # Add year column
  all_years_data[[as.character(year)]] <- df_year
  
  # Clear df_year to free up memory
  rm(df_year)
  gc()  # Trigger garbage collection
}

# Combine all the yearly data into a single data frame
combined_data <- bind_rows(all_years_data)

# Remove the list of chunks to free memory
rm(all_years_data)
gc()  # Trigger garbage collection

# Dropping missing values
combined_data <- na.omit(combined_data)

# Undersample the data before one-hot encoding
undersampled_data <- ovun.sample(Diverted ~ ., data = combined_data, method = "under", N = min(table(combined_data$Diverted)) * 2)$data

# Separate undersampled features and target
X_undersampled <- undersampled_data %>%
  select(CarrierName, Month)  # Select only the two features

y_undersampled <- as.numeric(undersampled_data$Diverted)  # Ensure target is numeric (0, 1)

# Remove undersampled_data from memory
rm(undersampled_data)
gc()  # Trigger garbage collection

# One-hot encoding for the two selected features (CarrierName and Month)
df_encoded_undersampled <- model.matrix(~ CarrierName + Month, X_undersampled)[, -1]  # Exclude intercept

# Define X and Y for logistic regression after undersampling
X_train <- df_encoded_undersampled
y_train <- as.numeric(ifelse(y_undersampled == "1", 1, 0))  # Ensure binary (0, 1) target

# Remove unused objects from memory
rm(X_undersampled, y_undersampled)
gc()  # Trigger garbage collection

# Train a logistic regression model
model <- glm(Diverted ~ ., data = as.data.frame(cbind(X_train, Diverted = y_train)), family = binomial)

# Predictions and probabilities
y_pred <- predict(model, newdata = as.data.frame(X_train), type = "response")
y_pred_class <- ifelse(y_pred > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(y_pred_class), as.factor(y_train))
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy, Precision, Recall, F1-Score
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
recall <- conf_matrix$byClass["Sensitivity"]        # Recall
f1 <- 2 * ((precision * recall) / (precision + recall))

cat("\nModel Evaluation Metrics:\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1, 4), "\n")

# Coefficients from the logistic regression model
coefficients <- coef(model)
features <- names(coefficients)

coef_df <- data.frame(
  Feature = features,
  Coefficient = coefficients
)

# Sort by absolute value of coefficient
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
print("Feature Coefficients:")
print(coef_df)

# Plot the Coefficients
ggplot(coef_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients", x = "Feature", y = "Coefficient")


# Close the database connection
dbDisconnect(conn)

# Clear any remaining objects and run garbage collection
rm(list = ls())
gc()
