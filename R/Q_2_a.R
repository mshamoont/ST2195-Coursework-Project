# Question 2 (a): What are the best times and days of the week to minimise delays each year?

#install.packages("tidyr")
# Load necessary libraries
library(DBI)
library(RSQLite)
library(dplyr)
library(ggplot2)
library(tidyr)
library(reshape2)


# Setting up range of the years
years <- 1996:2005

# Dictionary (list in R) to store results for each year
delay_summaries <- list()

# Database connection (update with your actual database file)
conn <- dbConnect(RSQLite::SQLite(), dbname = "/Users/muhammadshamoontariq/Desktop/LSE_Prog practice assignments/dataverse_files/flights_data.db")

# Loop through each year
for (year in years) {
  
  # Querying data for the specific year
  query <- paste0('SELECT DepTime, DayOfWeek, DepDelay FROM "', year, '";')
  df <- dbGetQuery(conn, query)
  
  # Convert DepTime and DepDelay to numeric, allowing for coercion warnings
  df <- df %>%
    mutate(
      DepTime = as.numeric(DepTime),
      DepDelay = as.numeric(DepDelay)
    ) %>%
    drop_na(DepTime, DepDelay)  # Drop rows with NA values after coercion
  
  # Proceed with the analysis as planned
  categorize_time <- function(time) {
    if (is.na(time)) {
      return('Unknown')
    } else if (0 <= time & time < 600) {
      return('Late Night')
    } else if (600 <= time & time < 1200) {
      return('Morning')
    } else if (1200 <= time & time < 1800) {
      return('Afternoon')
    } else {
      return('Evening')
    }
  }
  
  df$TimeCategory <- sapply(df$DepTime, categorize_time)
  
  delay_summary <- df %>%
    group_by(DayOfWeek, TimeCategory) %>%
    summarise(AvgDepDelay = mean(DepDelay, na.rm = TRUE)) %>%
    ungroup()
  
  # Adding the 'Year' column to the delay summary df
  delay_summary$Year <- year
  
  # Store the summary in the list
  delay_summaries[[as.character(year)]] <- delay_summary
  
  
}

# Prepare data for grouped bar chart by calculating the average delay for each time category and year
grouped_delay_summary <- combined_delay_summary %>%
  group_by(Year, TimeCategory) %>%
  summarise(AvgDepDelay = mean(AvgDepDelay, na.rm = TRUE)) %>%
  ungroup()

# Create a grouped bar chart with Year on the x-axis
ggplot(grouped_delay_summary, aes(x = factor(Year), y = AvgDepDelay, fill = TimeCategory)) +
  geom_bar(stat = "identity", position = "dodge") +  # Dodge to separate bars by TimeCategory
  labs(
    title = "Average Departure Delays by Year and Time Category",
    x = "Year",
    y = "Average Delay (minutes)",
    fill = "Time Category"
  ) +
  scale_fill_brewer(palette = "Paired") +  # Use a nice color palette for time categories
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability


# Close the database connection
dbDisconnect(conn)

warnings()


