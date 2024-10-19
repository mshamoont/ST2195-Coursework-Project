# Question 2 (b): Evaluate whether older planes suffer more delays on a year-to-year basis.

# Using only delay in Arrivals

# Load necessary libraries
library(DBI)
library(RSQLite)
library(dplyr)
library(ggplot2)

# Database connection (update with your actual database file path)
conn <- dbConnect(RSQLite::SQLite(), dbname = "/Users/muhammadshamoontariq/Desktop/LSE_Prog practice assignments/dataverse_files/flights_data.db")

# List of years from 1996 to 2005 for the analysis
years_of_analysis <- 1996:2005

# Initialize an empty data frame to hold combined data for all years
combined_df <- data.frame()

# Loop through each year
for (year_of_analysis in years_of_analysis) {
  
  # Query data for ArrDelay, TailNum, and ManufactureYear for the current year
  query <- paste0('
    SELECT f.ArrDelay, f.TailNum, p.Year AS ManufactureYear
    FROM "', year_of_analysis, '" f
    LEFT JOIN "plane-data" p ON f.TailNum = p.TailNum
    WHERE f.Year = ', year_of_analysis, ';')
  
  # Execute the query
  df <- dbGetQuery(conn, query)
  
  # Data Preprocessing: Convert columns to numeric and drop missing values
  df <- df %>%
    mutate(
      ArrDelay = as.numeric(ArrDelay),
      ManufactureYear = as.numeric(ManufactureYear)
    ) %>%
    drop_na(ArrDelay, ManufactureYear)
  
  # Filter out early arrivals (negative delays)
  df <- df %>%
    filter(ArrDelay > 0)
  
  # Filter out invalid values based on the manufacturing year
  df <- df %>%
    filter(ManufactureYear > 0, ManufactureYear < year_of_analysis)
  
  # Calculate Aircraft Age
  df$AircraftAge <- year_of_analysis - df$ManufactureYear
  
  # Take the absolute value of Arrival Delay
  df$LateArrDelay <- abs(df$ArrDelay)
  
  # Group by Aircraft Age and calculate average late arrival delay
  avg_abs_delay_by_age <- df %>%
    group_by(AircraftAge) %>%
    summarise(LateArrDelay = mean(LateArrDelay, na.rm = TRUE)) %>%
    ungroup()
  
  # Add current year to the data frame
  avg_abs_delay_by_age$YearOfAnalysis <- year_of_analysis
  
  # Append current year's data to the combined data frame
  combined_df <- bind_rows(combined_df, avg_abs_delay_by_age)
  
  # Plot the line chart for each year
  ggplot(avg_abs_delay_by_age, aes(x = AircraftAge, y = LateArrDelay)) +
    geom_line(linetype = "solid") +
    geom_point() +
    ggtitle(paste("Average Late Arrival Delay by Aircraft Age -", year_of_analysis)) +
    xlab("Aircraft Age (Years)") +
    ylab("Average Late Arrival Delay (Minutes)") +
    theme_minimal()
  
  # Scatter plot for each year
  ggplot(df, aes(x = AircraftAge, y = LateArrDelay)) +
    geom_point(alpha = 0.5, color = "blue") +
    ggtitle(paste("Scatter Plot: Aircraft Age vs Late Arrival Delay -", year_of_analysis)) +
    xlab("Aircraft Age (Years)") +
    ylab("Late Arrival Delay (Minutes)") +
    theme_minimal()
  
  # Print the correlation between Aircraft Age and Late Arrival Delay
  correlation <- cor(df$AircraftAge, df$LateArrDelay, use = "complete.obs")
  print(paste("Correlation between Aircraft Age and Late Arrival Delay for", year_of_analysis, ":", correlation))
}

# Create age groups for combined data
bins <- c(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
labels <- c('0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50')
combined_df$AgeGroup <- cut(combined_df$AircraftAge, breaks = bins, labels = labels, include.lowest = TRUE)

# Calculate average late arrival delay by age group
avg_abs_delay_by_age_group <- combined_df %>%
  group_by(AgeGroup) %>%
  summarise(LateArrDelay = mean(LateArrDelay, na.rm = TRUE))

print("Average Late Arrival Delay by Aircraft Age Group:")
print(avg_abs_delay_by_age_group)

# Plot combined results for all years
ggplot(combined_df, aes(x = AircraftAge, y = LateArrDelay, color = factor(YearOfAnalysis))) +
  geom_line() +
  geom_point() +
  ggtitle("Average Late Arrival Delay by Aircraft Age (1996-2005)") +
  xlab("Aircraft Age (Years)") +
  ylab("Average Late Arrival Delay (Minutes)") +
  theme_minimal() +
  theme(legend.title = element_text(face = "bold")) +
  guides(color = guide_legend(title = "Year"))

# Summary statistics and year-on-year comparison
summary_stats <- combined_df %>%
  group_by(YearOfAnalysis) %>%
  summarise(
    mean_delay = mean(LateArrDelay, na.rm = TRUE),
    median_delay = median(LateArrDelay, na.rm = TRUE),
    std_dev_delay = sd(LateArrDelay, na.rm = TRUE),
    min_delay = min(LateArrDelay, na.rm = TRUE),
    max_delay = max(LateArrDelay, na.rm = TRUE)
  )

print("Summary Statistics for Each Year:")
print(summary_stats)

# Calculate year-on-year change in mean delay
summary_stats <- summary_stats %>%
  mutate(YoY_Change_in_Mean_Delay = mean_delay - lag(mean_delay))

print("Year-on-Year Change in Mean Late Arrival Delay:")
print(summary_stats)

# Close the database connection
dbDisconnect(conn)