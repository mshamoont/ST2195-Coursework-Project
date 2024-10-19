#!/usr/bin/env python
# coding: utf-8

# # ST2195 Coursework Project

# # PART-1
# 
# In this part, We are asked to work with the Markov Chain Monte Carlo algorithm, in particular the Metropolis-Hastings algorithm. The aim is to simulate random numbers for the distribution with probability density function given below  <br>
# f(x) = 0.5 * exp(−|x|), <br>
# where x takes values in the real line and |x| denotes the absolute value of x. More specifically, you are asked to generate x0, x1, . . . , xN values and store them using the following version of the Metropolis-Hastings algorithm (also known as random walk Metropolis) that consists of the steps below:
# 

# ### Setting up Random Walk Metropolis

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


# Probability density function f(x)
# Target distribution f(x) = 1/2 * exp(-|x|)
def f(x):
    return 0.5 * np.exp(-np.abs(x))  

# Setting up initial values
x0 = 0  # Initial value of x
N = 10000  # Number of iterations
s = 1.0  # Standard deviation (step size) for the proposal distribution

# Array to store the generated values
x_values = np.zeros(N + 1)
x_values[0] = x0  

# Random walk Metropolis-Hastings
for i in range(1, N + 1):
    
    x_star = np.random.normal(x_values[i - 1], s)
    r = f(x_star) / f(x_values[i - 1])
    
    # u from Uniform(0, 1)
    u = np.random.uniform(0, 1)
    
    # Acceptance or Rejection criteria based on the ratio
    if u < r:
        x_values[i] = x_star  # Accept the new value
    else:
        x_values[i] = x_values[i - 1]  # Reject the new value, stay at the current value


# In[22]:


x_values


# ## Answer for Question 1.a
# <br>
# Apply the random walk Metropolis algorithm using N = 10000 and s = 1. Use the generated samples (x1, . . . xN ) to construct a histogram and a kernel density plot in the same figure. Note that these provide estimates of f (x).Overlay a graph of f (x) on this figure to visualise the quality of these estimates. Also, report the sample mean and standard deviation of the generated samples (Note: these are also known as the Monte Carlo estimates of the mean and standard deviation respectively).
# Practical tip: To avoid numerical errors, it is better to use the equivalent criterion log u < log r (x∗, xi−1) = log f (x∗) − log f (xi−1) instead of u < r (x∗, xi−1).

# In[23]:


# Random walk Metropolis-Hastings
for i in range(1, N + 1):
    
    x_star = np.random.normal(x_values[i - 1], s)
    
    # Ratio using log r(x*, x[i-1])
    log_r = np.log(f(x_star)) - np.log(f(x_values[i - 1]))   
    
    # u from Uniform(0, 1)
    u = np.random.uniform(0, 1)
    
    # # Acceptance or Rejection criteria based on the new ratio
    if np.log(u) < log_r:
        x_values[i] = x_star  # Accept the new value
    else:
        x_values[i] = x_values[i - 1]  # Reject the new value

# Sample mean and standard deviation
sample_mean = np.mean(x_values)
sample_std = np.std(x_values)


print(f"Sample Mean: {sample_mean}")
print(f"Sample Standard Deviation: {sample_std}")

# Plotting
plt.figure(figsize=(10, 6))

# Histogram and kernel density estimate
sns.histplot(x_values[1:], bins=50, stat="density", color='g', label='Histogram', kde=True)

# Overlay the target distribution
x = np.linspace(-10, 10, 1000)
plt.plot(x, f(x), 'r-', label='Target Distribution f(x)')


plt.xlabel('x')
plt.ylabel('Density')
plt.title('Metropolis-Hastings Sampling')
plt.legend()
plt.grid()
plt.show()


# ---
# ---
# ---
# ---
# ---

# ## Answer for Question 1.b
# <br>
# The operations in part 1(a) are based on the assumption that the algorithm has converged. One of the most widely used convergence diagnostics is the so-called Rb value. In order to obtain a valued of this diagnostic, you need to apply the procedure below:
# • Generate more than one sequence of x0,...,xN, potentially using different
# initial values x0. Denote each of these sequences, also known as chains, by
# (x(j),x(j),...,x(j)) for j = 1,2,...,J. 01N
# • Define and compute Mj as the sample mean of chain j as 1N
# i=1
# and Vj as the within sample variance of chain j as
# Mj = Xx(j). Ni
# 1N
# Vj = X(x(j) −Mj)2.
# Ni i=1
# • Define and compute the overall within sample variance W as 1 XJ
# W=J Vj j=1
# • Define and compute the overall sample mean M as 1 XJ
# M=J Mj, j=1
# and the between sample variance B as 1 XJ
# • Compute the Rb value as
# In general, values of Rb close to 1 indicate convergence, and it is usually desired for Rb to be lower than 1.05. Calculate the Rb for the random walk Metropolis algorithm with N = 2000, s = 0.001 and J = 4. Keeping N and J fixed, provide a plot of the values of Rb over a grid of s values in the interval between 0.001 and 1.

# In[25]:


# Target distribution function f(x)
def f(x):
    return 0.5 * np.exp(-np.abs(x))

# Metropolis-Hastings algorithm for a single chain
def metropolis_hastings_chain(x0, N, s, burn_in=500):
    x_values = np.zeros(N + burn_in)  
    x_values[0] = x0
    for i in range(1, N + burn_in):
        x_star = np.random.normal(x_values[i-1], s)
        log_r = np.log(f(x_star)) - np.log(f(x_values[i-1]))
        u = np.random.uniform(0, 1)
        if np.log(u) < log_r:
            x_values[i] = x_star
        else:
            x_values[i] = x_values[i-1]
    
    # Chain after the burn-in period
    return x_values[burn_in:]

# R_hat value
def compute_r_hat(chains):
    J = len(chains)  
    N = len(chains[0])  
    
    #  Mj(mean of each chain) and Vj(variance of each chain)
    Mj = np.mean(chains, axis=1)
    Vj = np.var(chains, axis=1, ddof=1)
    
    # Overall mean M
    M = np.mean(Mj)
    
    # Within-sample variance W
    W = np.mean(Vj)
    
    # Between-sample variance B
    B = np.mean((Mj - M)**2)
    
    # R_hat
    R_hat = np.sqrt((B + W) / W)
    return R_hat

# Multiple chains and R_hat over a grid of s values
N = 2000  
J = 4  
burn_in = 500  
initial_values = [0, 1, -1, 2]  
s_values = np.linspace(0.01, 1, 100)  
r_hat_values = []

# Cap for R_hat values
r_hat_cap = 2  

for s in s_values:
    chains = []
    for x0 in initial_values:
        chain = metropolis_hastings_chain(x0, N, s, burn_in=burn_in)
        chains.append(chain)
    chains = np.array(chains)
    r_hat = compute_r_hat(chains)
    
    # Capped R_hat value for visualization
    if r_hat > r_hat_cap:
        r_hat = r_hat_cap  # Cap the R_hat value
    r_hat_values.append(r_hat)

# Plotting
plt.plot(s_values, r_hat_values, label=r'$\hat{R}$ (capped)')
plt.axhline(y=1.05, color='r', linestyle='--', label='Threshold 1.05')
plt.xlabel('Step size (s)')
plt.ylabel(r'$\hat{R}$ value')
plt.title(r'$\hat{R}$ values for different step sizes s (capped)')
plt.legend()
plt.grid(True)
plt.show()


# ---
# ---
# ---
# ---
# ---

# # PART-2

# #### The 2009 ASA Statistical Computing and Graphics Data Expo consisted of flight arrival and departure details for all commercial flights on major carriers within the USA from Oc- tober 1987 to April 2008. This is a large dataset; there are nearly 120 million records in total, and it takes up 1.6 gigabytes of space when compressed and 12 gigabytes when un- compressed. The complete dataset, along with supplementary information and variable descriptions, can be downloaded from the Harvard Dataverse at https://doi.org/10.7910/DVN/HG7NV7
# Choose any subset of ten consecutive years and any of the supplementary information provided by the Harvard Dataverse to answer the following questions using the principles and tools you have learned in this course:
# 

# ## I am choosing 10 years period of 1996-2005 

# In[19]:


# Uploading Downloaded flights data to SQL

import os
import sqlite3
import pandas as pd


csv_folder_path = "/Users/muhammadshamoontariq/Desktop/LSE_Prog practice assignments/dataverse_files/Analysis_data_csv"  # Your folder path with CSV files
db_file_path = "flights_data.db"  # Path to your SQLite database file

# Connecting to SQL
conn = sqlite3.connect(db_file_path)

# Looping through 14 files
for file in os.listdir(csv_folder_path):
    if file.endswith(".csv"):  
        file_path = os.path.join(csv_folder_path, file)
        table_name = os.path.splitext(file)[0]

        print(f"Processing file: {file}")  

        try:
            # Encountered encoding error so add exception
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')

        # Uploading the DataFrame to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Uploaded {file} to the '{table_name}' table.")


conn.close()
print("All CSV files have been uploaded.")


# ## Answer to Question 2 a)
# 
# ### What are the best times and days of the week to minimise delays each year?
# 

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3


# In[27]:


#Connecting to the SQLite database
conn = sqlite3.connect('flights_data.db')  # Update with your actual database name


# #### First anlyzing only one year (2005)

# In[28]:


#inspecting data
year = '2005'
query = f'SELECT * FROM "{year}";'
flights_2005 = pd.read_sql_query(query, conn)
flights_2005.head(50)


# In[29]:


# Fetching data only for Departure time, Day of the week & Departure Delays
query_1 = f'SELECT DepTime, DayOfWeek, DepDelay FROM "{year}";'
df = pd.read_sql_query(query_1, conn)


# In[30]:


#Preprocessing
# Converting object data type to Numeric AND removing NA values
df['DepTime'] = pd.to_numeric(df['DepTime'], errors='coerce')
df['DepDelay'] = pd.to_numeric(df['DepDelay'], errors='coerce')
df.dropna(inplace=True)  # Remove NA values


# In[31]:


# Converting DepTime into time categories (e.g., morning, afternoon, evening)
# Adding a new column 'TimeCategory' based on DepTime ranges
def categorize_time(time):
    if pd.isnull(time):
        return 'Unknown'
    time = int(time)
    if 0 <= time < 600:
        return 'Late Night'
    elif 600 <= time < 1200:
        return 'Morning'
    elif 1200 <= time < 1800:
        return 'Afternoon'
    else:
        return 'Evening'

df['TimeCategory'] = df['DepTime'].apply(categorize_time)


# In[32]:


# Group by DayOfWeek and TimeCategory, and calculate the average departure delay
delay_summary = df.groupby(['DayOfWeek', 'TimeCategory'])['DepDelay'].mean().reset_index()


# Pivot table to create a heatmap-friendly format
pivot_df = delay_summary.pivot(index='DayOfWeek', columns='TimeCategory', values='DepDelay')


# In[34]:


# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Average Departure Delay by Day of Week and Time of Day')
plt.ylabel('Day of Week')
plt.xlabel('Time of Day')
plt.show()


# # 10-year period (1996 to 2005) Analysis
# 
# ### Best times and days of the week to minimise delays each year?

# In[35]:


#Setting up range of the years
years = [str(year) for year in range(1996, 2006)]  

# dictionary to store results for each year
delay_summaries = {}

for year in years:
    
    query = f'SELECT DepTime, DayOfWeek, DepDelay FROM "{year}";'
    df = pd.read_sql_query(query, conn)

   
    df['DepTime'] = pd.to_numeric(df['DepTime'], errors='coerce')
    df['DepDelay'] = pd.to_numeric(df['DepDelay'], errors='coerce')
    df.dropna(inplace=True)  # Remove NA values


    # Converting DepTime into time categories (e.g., morning, afternoon, evening)
    def categorize_time(time):
        if pd.isnull(time):
            return 'Unknown'
        time = int(time)
        if 0 <= time < 600:
            return 'Late Night'
        elif 600 <= time < 1200:
            return 'Morning'
        elif 1200 <= time < 1800:
            return 'Afternoon'
        else:
            return 'Evening'

    df['TimeCategory'] = df['DepTime'].apply(categorize_time)

    # Group by DayOfWeek and TimeCategory, and calculate the average departure delay
    delay_summary = df.groupby(['DayOfWeek', 'TimeCategory'])['DepDelay'].mean().reset_index()

    # Adding the 'Year' column to the delay summary df
    delay_summary['Year'] = year

    delay_summaries[year] = delay_summary

    # Pivot table to create a heatmap-friendly format
    pivot_df = delay_summary.pivot(index='DayOfWeek', columns='TimeCategory', values='DepDelay')

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title(f'Average Departure Delay by Day of Week and Time of Day for {year}')
    plt.ylabel('Day of Week (1=Mon, 7=Sun)')
    plt.xlabel('Time of Day')
    plt.show()

# Combining all the delay summaries into a single DataFrame for further analysis
combined_delay_summary = pd.concat(delay_summaries.values(), ignore_index=True)


# In[36]:


combined_delay_summary


# In[37]:


# average delays for each year
avg_delays_per_year = {year: delay_summary['DepDelay'].mean() for year, delay_summary in delay_summaries.items()}

plt.figure(figsize=(10, 6))
plt.bar(avg_delays_per_year.keys(), avg_delays_per_year.values(), color='skyblue')
plt.title('Average Departure Delays by Year')
plt.xlabel('Year')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45)
plt.show()

avg_delays_per_year_df = pd.DataFrame(list(avg_delays_per_year.items()), columns=['Year', 'Average_DepDelay'])


# In[38]:


avg_delays_per_year_df


# In[39]:


# Grouped bar chart
grouped_delay_summary = {}
for year, summary in delay_summaries.items():
    for time_cat in summary['TimeCategory'].unique():
        avg_delay = summary[summary['TimeCategory'] == time_cat]['DepDelay'].mean()
        grouped_delay_summary.setdefault(time_cat, {})[year] = avg_delay

# plotting
grouped_delay_df = pd.DataFrame(grouped_delay_summary).fillna(0)

grouped_delay_df.plot(kind='bar', figsize=(12, 6))
plt.title('Average Departure Delays by Time Category and Year')
plt.xlabel('Time Category')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.show()


# In[40]:


grouped_delay_df


# In[41]:


# Minimum delays by DayOfWeek and TimeCategory for each year
optimal_times = {}

for year, summary in delay_summaries.items():
    min_delay = summary.loc[summary['DepDelay'].idxmin()]
    optimal_times[year] = {
        'Best Day': min_delay['DayOfWeek'],
        'Best Time Category': min_delay['TimeCategory'],
        'Min Delay': min_delay['DepDelay']
    }

optimal_times_df = pd.DataFrame(optimal_times).T
print("Optimal Times and Days for Each Year:")
optimal_times_df


# In[42]:


# Maximum delay by DayOfWeek and TimeCategory for each year
max_delay_times = {}

for year, summary in delay_summaries.items():
    max_delay = summary.loc[summary['DepDelay'].idxmax()]  # Find the row with the max average delay
    max_delay_times[year] = {
        'Worst Day': max_delay['DayOfWeek'],
        'Worst Time Category': max_delay['TimeCategory'],
        'Max Delay': max_delay['DepDelay']
    }

max_delay_df = pd.DataFrame(max_delay_times).T
print("Maximum Delays for Each Year:")
max_delay_df


# ---
# ---
# ---
# ---
# ---

# # Answer to Question 2 b)
# 
# ### Evaluate whether older planes suffer more delays on a year-to-year basis?

# #### First Analyzing only one year (2005)

# In[43]:


year_of_analysis = 2005  

# Getting Arrival delays and Tail-number from flights data table and joining on Tail-Number with Planes table to get Year of Manufacture  
query = '''
SELECT f.ArrDelay, f.TailNum, p.Year as ManufactureYear
FROM "2005" f  
LEFT JOIN "plane-data" p ON f.TailNum = p.TailNum
WHERE f.Year = ?;  
'''

# Execute the query with the year_of_analysis parameter
df_age = pd.read_sql_query(query, conn, params=(year_of_analysis,))
df_age.head(50)


# In[44]:


#Data Preprocessing
#Converting Object type to Numeric
df_age['ArrDelay'] = pd.to_numeric(df_age['ArrDelay'], errors='coerce')  
df_age['ManufactureYear'] = pd.to_numeric(df_age['ManufactureYear'], errors='coerce')  

# Removing NA rows (Also include 0 values as some fields are missing data)
df_age.dropna(subset=['ArrDelay', 'ManufactureYear'], inplace=True)  

#Counting invalid values (Removing planes which were provided same year or after the year under anlysis)

same_year_count = (df_age['ManufactureYear'] == year_of_analysis).sum()
beyond_year_count = (df_age['ManufactureYear'] > year_of_analysis).sum()

# Total count of invalid values (not strictly necessary for removal)
total_invalid_count = same_year_count + beyond_year_count


# indices of rows to drop
indices_to_drop = df_age[
    (df_age['ManufactureYear'] == year_of_analysis) |  # Same year
    (df_age['ManufactureYear'] > year_of_analysis) |   # Beyond year of analysis
    (df_age['ManufactureYear'] == 0)                    # Zero values (Some data is missing in the provided database)
].index

df_age.drop(indices_to_drop, inplace=True)


# In[45]:


# Counting how many values in 'year' are null
null_count = df_age['ManufactureYear'].isnull().sum()

# Counting how many values in 'year' are empty strings
empty_count = (df_age['ManufactureYear'] == "").sum()

# Counting how many values in 'year' are zeero
zero_count = (df_age['ManufactureYear'] == 0).sum()


# Counting how many values in 'year' are 2005
same_year_count = (df_age['ManufactureYear'] == 2005).sum()

# Counting how many values in 'year' are greater than 2005
beyond_year_count = (df_age['ManufactureYear'] > 2005).sum()


# Total count of null, empty, and zero values
total_invalid_count = null_count + empty_count + same_year_count + beyond_year_count + zero_count 

print(f"Number of null values in 'year': {null_count}")
print(f"Number of empty values in 'year': {empty_count}")
print(f"Number of zero values in 'year': {zero_count}")

print(f"Number of same year in 'year': {same_year_count}")
print(f"Number of beyond year in 'year': {beyond_year_count}")
print(f"Total number of null, empty, or zero values in 'year': {total_invalid_count}")


# In[46]:


# Inspecting data if any invalid entries exists. Group by ManufactureYear and count occurrences
year_counts = df_age['ManufactureYear'].value_counts().sort_index()

# Plotting the bar chart
plt.figure(figsize=(12, 6))  # Set the figure size
year_counts.plot(kind='bar', color='skyblue', edgecolor='black')


plt.title('Number of Aircraft by Manufacture Year')
plt.xlabel('Manufacture Year')
plt.ylabel('Number of Aircraft')


for index, value in enumerate(year_counts):
    plt.text(index, value, str(value), ha='center', va='bottom')

# Show the plot
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()


# In[47]:


#Calculating Aircraft Age & Delays
# flight year is 2005 for all rows 
df_age['AircraftAge'] = 2005 - df_age['ManufactureYear']

# Step 4: Group by AircraftAge and calculate average arrival delay
avg_delay_by_age = df_age.groupby('AircraftAge')['ArrDelay'].mean().reset_index()

Delay= avg_delay_by_age.sort_values(by = 'AircraftAge')
Delay


# In[48]:


#Visualizing through line chart
plt.figure(figsize=(10, 6))
plt.plot(avg_delay_by_age['AircraftAge'], avg_delay_by_age['ArrDelay'], linestyle='-')


plt.title('Arrival Delay by Aircraft Age')
plt.xlabel('Aircraft Age (Years)')
plt.ylabel('Arrival Delay (Minutes)')

plt.grid(True)
plt.show()


# In[49]:


#Visualizing through scatter plot

# For example, let's collect all delays for each aircraft age
delays_by_age = df_age.groupby('AircraftAge')['ArrDelay'].apply(list).reset_index()


## Step 3: Plot the results
plt.figure(figsize=(12, 6))

# Scatter plot for each age
for index, row in delays_by_age.iterrows():
    plt.scatter([row['AircraftAge']] * len(row['ArrDelay']), row['ArrDelay'], alpha=0.6)

# Setting the title and labels
plt.title('Arrival Delays Grouped by Aircraft Age')
plt.xlabel('Aircraft Age (Years)')
plt.ylabel('Arrival Delay (Minutes)')

# Setting x-ticks to be integers
plt.xticks(range(int(df_age['AircraftAge'].min()), int(df_age['AircraftAge'].max() + 1)))  

plt.grid()
plt.show()


# In[50]:


delays_by_age


# In[ ]:





# # 10-year period (1996 to 2005) Analysis
# 
# ### Evaluate whether older planes suffer more delays on a year-to-year basis?

# In[51]:


#INCLUDES "DELAYS" AND "EARLY ARRIVALS"

# Average ABSOLUTE DELAYS.                                                                                                                                                                                  


# List of years from 1996 to 2005 for the analysis
years_of_analysis = range(1996, 2006)

# Initialize an empty DataFrame to hold combined data for all years
combined_df = pd.DataFrame()

# Loop through each year
for year_of_analysis in years_of_analysis:
    # Step 1: Query only ArrDelay and TailNum from the current year and join with ManufactureYear
    query = f'''
    SELECT f.ArrDelay, f.TailNum, p.Year as ManufactureYear
    FROM "{year_of_analysis}" f  
    LEFT JOIN "plane-data" p ON f.TailNum = p.TailNum
    WHERE f.Year = ?;  
    '''
    
    # Execute the query with the year_of_analysis parameter
    df = pd.read_sql_query(query, conn, params=(year_of_analysis,))

    # Step 2: Data Preprocessing
    df['ArrDelay'] = pd.to_numeric(df['ArrDelay'], errors='coerce')  # Ensure ArrDelay is numeric
    df['ManufactureYear'] = pd.to_numeric(df['ManufactureYear'], errors='coerce')  # Convert ManufactureYear to numeric

    df.dropna(subset=['ArrDelay', 'ManufactureYear'], inplace=True)  # Remove rows with missing

    # Step 3: Count invalid values
    same_year_count = (df['ManufactureYear'] == year_of_analysis).sum()
    beyond_year_count = (df['ManufactureYear'] > year_of_analysis).sum()

    # Identify indices of rows to drop
    indices_to_drop = df[
        (df['ManufactureYear'] == year_of_analysis) |  # Same year
        (df['ManufactureYear'] > year_of_analysis) |   # Beyond year of analysis
        (df['ManufactureYear'] == 0)                   # Zero values
    ].index

    # Drop those indices from the DataFrame
    df.drop(indices_to_drop, inplace=True)

    # Step 4: Calculate Aircraft Age
    df['AircraftAge'] = year_of_analysis - df['ManufactureYear']

    # Step 5: Take absolute value of Arrival Delay
    df['AbsArrDelay'] = df['ArrDelay'].abs()

    # Step 6: Group by AircraftAge and calculate average absolute arrival delay
    avg_abs_delay_by_age = df.groupby('AircraftAge')['AbsArrDelay'].mean().reset_index()

    # Add current year to the DataFrame
    avg_abs_delay_by_age['YearOfAnalysis'] = year_of_analysis

    # Append the current year's data to the combined DataFrame
    combined_df = pd.concat([combined_df, avg_abs_delay_by_age], ignore_index=True)

    # Step 7: Plot the line chart for each year
    plt.figure(figsize=(10, 6))
    plt.plot(avg_abs_delay_by_age['AircraftAge'], avg_abs_delay_by_age['AbsArrDelay'], linestyle='-', marker='o', label=f'{year_of_analysis}')

    # Step 8: Customize the plot for each year
    plt.title(f'Average Absolute Arrival Delay by Aircraft Age - {year_of_analysis}')
    plt.xlabel('Aircraft Age (Years)')
    plt.ylabel('Average Absolute Arrival Delay (Minutes)')
    plt.grid(True)
    plt.legend()

    # Step 9: Plot scatter plot for the current year
    plt.figure(figsize=(10, 6))
    plt.scatter(df['AircraftAge'], df['AbsArrDelay'], alpha=0.5, c='blue')
    plt.title(f'Scatter Plot: Aircraft Age vs Absolute Arrival Delay - {year_of_analysis}')
    plt.xlabel('Aircraft Age (Years)')
    plt.ylabel('Absolute Arrival Delay (Minutes)')
    plt.grid(True)
    
    # Show both plots
    plt.show()

    # Step 10: Calculate and print correlation between Aircraft Age and Arrival Delay
    correlation = df['AircraftAge'].corr(df['AbsArrDelay'])
    print(f"Correlation between Aircraft Age and Absolute Arrival Delay for {year_of_analysis}: {correlation}")

    # Step 11: Create age groups for combined data
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Define bins for age groups
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '45-50']
    combined_df['AgeGroup'] = pd.cut(combined_df['AircraftAge'], bins=bins, labels=labels)

    # Calculate average absolute delay by age group
    avg_abs_delay_by_age_group = combined_df.groupby('AgeGroup')['AbsArrDelay'].mean().reset_index()
    print("Average Absolute Arrival Delay by Aircraft Age Group:")
    print(avg_abs_delay_by_age_group)

# Step 12: Plot the combined results
plt.figure(figsize=(10, 6))

# Iterate over each year and plot the average absolute delay by aircraft age
for year in years_of_analysis:
    yearly_data = combined_df[combined_df['YearOfAnalysis'] == year]
    plt.plot(yearly_data['AircraftAge'], yearly_data['AbsArrDelay'], label=f'Year {year}', marker='o', linestyle='-')

# Step 13: Customize the plot
plt.title('Average Absolute Arrival Delay by Aircraft Age (1996-2005)')
plt.xlabel('Aircraft Age (Years)')
plt.ylabel('Average Absolute Arrival Delay (Minutes)')
plt.legend(title='Year')
plt.grid(True)

# Display the plot
plt.show()

# Summary statistics and year-on-year comparison
summary_stats = combined_df.groupby('YearOfAnalysis').agg(
    mean_delay=('AbsArrDelay', 'mean'),
    median_delay=('AbsArrDelay', 'median'),
    std_dev_delay=('AbsArrDelay', 'std'),
    min_delay=('AbsArrDelay', 'min'),
    max_delay=('AbsArrDelay', 'max')
).reset_index()

print("Summary Statistics for Each Year:")
print(summary_stats)

# Calculate year-on-year change in mean delay
summary_stats['YoY Change in Mean Delay'] = summary_stats['mean_delay'].diff()
print("\nYear-on-Year Change in Mean Absolute Delay:")
print(summary_stats[['YearOfAnalysis', 'YoY Change in Mean Delay']])


# In[52]:


yearly_data


# In[ ]:





# ### Removing Early Arrivals from the  Delays (Negative Values) 
# ### For Meaningful Analysis

# In[53]:


#ONLY Delays (Not early Arrivals)

# List of years from 1996 to 2005 for the analysis
years_of_analysis = range(1996, 2006)

# Empty DataFrame to hold combined data for all years
combined_df = pd.DataFrame()

# Looping through each year
for year_of_analysis in years_of_analysis:
    
    query = f'''
    SELECT f.ArrDelay, f.TailNum, p.Year as ManufactureYear
    FROM "{year_of_analysis}" f  
    LEFT JOIN "plane-data" p ON f.TailNum = p.TailNum
    WHERE f.Year = ?;  
    '''
    
    
    df = pd.read_sql_query(query, conn, params=(year_of_analysis,))

    # Data Preprocessing
    df['ArrDelay'] = pd.to_numeric(df['ArrDelay'], errors='coerce')  # Ensure ArrDelay is numeric
    df['ManufactureYear'] = pd.to_numeric(df['ManufactureYear'], errors='coerce')  # Convert ManufactureYear to numeric

    # Removing early arrivals (negative ArrDelay)
    df = df[df['ArrDelay'] > 0]  

    # Dropping rows with missing data 
    df.dropna(subset=['ArrDelay', 'ManufactureYear'], inplace=True)

   
    same_year_count = (df['ManufactureYear'] == year_of_analysis).sum()
    beyond_year_count = (df['ManufactureYear'] > year_of_analysis).sum()

   
    indices_to_drop = df[
        (df['ManufactureYear'] == year_of_analysis) |  # Same year
        (df['ManufactureYear'] > year_of_analysis) |   # Beyond year of analysis
        (df['ManufactureYear'] == 0)                   # Zero values
    ].index

    
    df.drop(indices_to_drop, inplace=True)

    # S Aircraft Age
    df['AircraftAge'] = year_of_analysis - df['ManufactureYear']

    # Taking absolute value of Arrival Delay (only positive delays remain)
    df['LateArrDelay'] = df['ArrDelay'].abs()

    # Group by AircraftAge and calculate average absolute arrival delay
    avg_abs_delay_by_age = df.groupby('AircraftAge')['LateArrDelay'].mean().reset_index()

    # Adding current year to the DataFrame
    avg_abs_delay_by_age['YearOfAnalysis'] = year_of_analysis

    # Appending the current year's data to the combined DataFrame
    combined_df = pd.concat([combined_df, avg_abs_delay_by_age], ignore_index=True)

    # Plotting the line chart for each year
    plt.figure(figsize=(10, 6))
    plt.plot(avg_abs_delay_by_age['AircraftAge'], avg_abs_delay_by_age['LateArrDelay'], linestyle='-', marker='o', label=f'{year_of_analysis}')


    plt.title(f'Average Late Arrival Delay by Aircraft Age - {year_of_analysis}')
    plt.xlabel('Aircraft Age (Years)')
    plt.ylabel('Average Late Arrival Delay (Minutes)')
    plt.grid(True)
    plt.legend()

    # Plotting scatter plot for the current year
    plt.figure(figsize=(10, 6))
    plt.scatter(df['AircraftAge'], df['LateArrDelay'], alpha=0.5, c='blue')
    plt.title(f'Scatter Plot: Aircraft Age vs Late Arrival Delay - {year_of_analysis}')
    plt.xlabel('Aircraft Age (Years)')
    plt.ylabel('Late Arrival Delay (Minutes)')
    plt.grid(True)
    
   
    plt.show()

    #  Statistical analysis between Aircraft Age and Absolute Arrival Delay
    correlation = df['AircraftAge'].corr(df['LateArrDelay'])
    print(f"Correlation between Aircraft Age and LateArrDelay Arrival Delay for {year_of_analysis}: {correlation}")

    # Creating age groups for combined data
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Define bins for age groups
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '45-50']
    combined_df['AgeGroup'] = pd.cut(combined_df['AircraftAge'], bins=bins, labels=labels)

    # Calculating average absolute delay by age group
    avg_abs_delay_by_age_group = combined_df.groupby('AgeGroup')['LateArrDelay'].mean().reset_index()
    print("Average Late Arrival Delay by Aircraft Age Group:")
    print(avg_abs_delay_by_age_group)

# Plotting the combined results
plt.figure(figsize=(10, 6))


for year in years_of_analysis:
    yearly_data_delays = combined_df[combined_df['YearOfAnalysis'] == year]
    plt.plot(yearly_data_delays['AircraftAge'], yearly_data_delays['LateArrDelay'], label=f'Year {year}', marker='o', linestyle='-')


plt.title('Average Late Arrival Delay by Aircraft Age (1996-2005)')
plt.xlabel('Aircraft Age (Years)')
plt.ylabel('Average Late Arrival Delay (Minutes)')
plt.legend(title='Year')
plt.grid(True)


plt.show()

# Summary statistics and Y/Y comparison
summary_stats = combined_df.groupby('YearOfAnalysis').agg(
    mean_delay=('LateArrDelay', 'mean'),
    median_delay=('LateArrDelay', 'median'),
    std_dev_delay=('LateArrDelay', 'std'),
    min_delay=('LateArrDelay', 'min'),
    max_delay=('LateArrDelay', 'max')
).reset_index()

print("Summary Statistics for Each Year:")
print(summary_stats)

# Y/Y change in mean delay
summary_stats['YoY Change in Mean Delay'] = summary_stats['mean_delay'].diff()
print("\nYear-on-Year Change in Mean Absolute Delay:")
print(summary_stats[['YearOfAnalysis', 'YoY Change in Mean Delay']])


# In[54]:


summary_stats


# In[55]:


yearly_data_delays


# In[100]:


yearly_data_delays.to_csv('yearly_data_delays.csv', index = False)


# ---
# ---
# ---
# ---
# ---

# # Answer to Question 2 c)  
# 
# ### For each year, fit a logistic regression model for the probability of diverted US flights using as many features as possible from attributes of the departure date, the sched- uled departure and arrival times, the coordinates and distance between departure and planned arrival airports, and the carrier. Visualize the coefficients across years.

# In[56]:


import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    classification_report, 
    confusion_matrix
)


# ### Analyzing one year first (1998)

# In[57]:


# Querying the relevant features from the 1998 table and joining with the carriers table
query = '''
SELECT 
    f.Month, 
    f.DayofMonth,  
    f.CRSArrTime AS ScheduledArrival, 
    f.CRSDepTime AS ScheduledDeparture, 
    f.Distance,       
    f.Diverted, 
    c.Description AS CarrierName,
    a1.lat AS Origin_Latitude,
    a1.long AS Origin_Longitude,
    a2.lat AS Dest_Latitude,
    a2.long AS Dest_Longitude
FROM 
    "1998" f 
LEFT JOIN 
    "carriers" c ON f.UniqueCarrier = c.Code
LEFT JOIN 
    "airports" a1 ON f.Origin = a1.iata 
LEFT JOIN 
    "airports" a2 ON f.Dest = a2.iata   
WHERE 
    f.Diverted IS NOT NULL;
'''


df_1998 = pd.read_sql_query(query, conn)


# In[45]:


df_1998.head(50)


# In[46]:


df_1998.info()


# In[47]:


df_1998.isna().sum()


# In[58]:


# Converting categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df_1998, columns=['CarrierName', 'Month', 'DayofMonth'], drop_first=True)

# Defining X and Y
X = df_encoded.drop(columns=['Diverted'])
y = df_encoded['Diverted']

# Splittiing the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fitting the logistic regression model
model = LogisticRegression(max_iter=1000, class_weight = 'balanced')
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

# Calculating accuracy, generate confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)

# Model coefficients
coefficients = model.coef_[0]
features = X.columns
for feature, coeff in zip(features, coefficients):
    print(f"Feature: {feature}, Coefficient: {coeff:.4f}")


# In[60]:


# Creating a DataFrame for coefficients and features
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
})


# Absolute values for sorting
coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])

# Absolute coefficient in descending order
sorted_coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Top 20 highest coefficients
print("\nTop 20 Features by Absolute Coefficient:")
sorted_coef_df.head(20)


# ## The model performs poorly in predicting diverted flights due to class imbalance. 
# 

# # Trying with Reduced features for year 1998 
# 
# ### based on selecting features from top 20 coefficients 

# In[61]:


#Query 2 with less features

# Querying the relevant features from the 1998 table and joining with the carriers table
query_2 = '''
SELECT 
    f.Month, 
    f.DayofMonth,        
    f.Diverted, 
    c.Description AS CarrierName,
    a1.lat AS Origin_Latitude,
    a1.long AS Origin_Longitude,
    a2.lat AS Dest_Latitude,
    a2.long AS Dest_Longitude
FROM 
    "1998" f 
LEFT JOIN 
    "carriers" c ON f.UniqueCarrier = c.Code
LEFT JOIN 
    "airports" a1 ON f.Origin = a1.iata 
LEFT JOIN 
    "airports" a2 ON f.Dest = a2.iata   
WHERE 
    f.Diverted IS NOT NULL;
'''


df_1998 = pd.read_sql_query(query_2, conn)

df_1998.dropna(inplace = True)

# Counting the number of diverted and non-diverted flights
diverted_counts = df_1998['Diverted'].value_counts()


print("Number of non-diverted flights (0):", diverted_counts.get(0, 0))
print("Number of diverted flights (1):", diverted_counts.get(1, 0))


# In[62]:


# Converting categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df_1998, columns=['CarrierName', 'Month', 'DayofMonth'], drop_first=True)

# Defining X and Y
X = df_encoded.drop('Diverted', axis=1)
y = df_encoded['Diverted']

# Splittiing the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fitting the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Calculating coefficients and odds ratios
coefficients = model.coef_[0]
features = X.columns
odds_ratios = np.exp(coefficients)


coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios
})

# Absolute values for sorting
coef_df['Abs_Coefficient'] = np.abs(coefficients)

# Highest and lowest impact variables
highest_impact = coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(1)
lowest_impact = coef_df.sort_values(by='Abs_Coefficient', ascending=True).head(1)


print("Variable with Highest Impact:")
print(highest_impact[['Feature', 'Coefficient', 'Odds_Ratio']])

print("\nVariable with Least Impact:")
print(lowest_impact[['Feature', 'Coefficient', 'Odds_Ratio']])

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ## The model performs poorly again in predicting diverted flights due to class imbalance. 
# <br> Sampling (Under or Over) should be used. Also for such datasets, Logistics Regression might not be the best model (Other models are out of scope of this Assignment)

# ## Need to Perform Undersampling to balance the data for Diversions and non-Diversion
# <br>
# As reducing features is not adding meaning to the analysis due to highly skewed data

# In[63]:


#Querying the relevant features (similar to previous query)
query_undersample = '''
SELECT 
    f.Month, 
    f.DayofMonth,        
    f.Diverted, 
    c.Description AS CarrierName,
    a1.lat AS Origin_Latitude,
    a1.long AS Origin_Longitude,
    a2.lat AS Dest_Latitude,
    a2.long AS Dest_Longitude
FROM 
    "1998" f 
LEFT JOIN 
    "carriers" c ON f.UniqueCarrier = c.Code
LEFT JOIN 
    "airports" a1 ON f.Origin = a1.iata 
LEFT JOIN 
    "airports" a2 ON f.Dest = a2.iata   
WHERE 
    f.Diverted IS NOT NULL;
'''


df_1998 = pd.read_sql_query(query_undersample, conn)


# In[63]:


df_encoded = pd.get_dummies(df_1998, drop_first=True)

X = df_encoded.drop('Diverted', axis=1)
y = df_encoded['Diverted']

# Applying Random Undersampling to balance the classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# New class distribution
print("Class distribution after undersampling:", y_resampled.value_counts())


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

coefficients = model.coef_[0]
features = X.columns
odds_ratios = np.exp(coefficients)


coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios
})


coef_df['Abs_Coefficient'] = np.abs(coefficients)


highest_impact = coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(1)
lowest_impact = coef_df.sort_values(by='Abs_Coefficient', ascending=True).head(1)


print("Variable with Highest Impact:")
print(highest_impact[['Feature', 'Coefficient', 'Odds_Ratio']])

print("\nVariable with Least Impact:")
print(lowest_impact[['Feature', 'Coefficient', 'Odds_Ratio']])


y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ### With undersampling, some meaning to analysis is achieved, however huge dataset is lost and is not the true reflection of the features impacting diversion. 
# 
# ### Logistics regression might not be the right model for predicting diversions. However, other modeling techniques are out of scope for this assignment.

# ---
# ---
# 

# # Analyzing 10-years period (1996-2005)   
# 
# ### Seems Undersampling is better way to go with the diversion analysis as data is heavily skewed towards non-diversions

# In[102]:


# Range of years for the analysis
years_of_analysis = list(range(1996, 2006))

# List to hold data for all years
all_years_data = []

# Querying Data in a Loop
for year in years_of_analysis:
    print(f"Querying data for the year: {year}")

    # Querying the data for the specific year
    query = f'''
    SELECT 
        f.Month, 
        f.DayofMonth,        
        f.Diverted, 
        f.CRSArrTime AS ScheduledArrival, 
        f.CRSDepTime AS ScheduledDeparture, 
        f.Distance,
        c.Description AS CarrierName,
        a1.lat AS Origin_Latitude,
        a1.long AS Origin_Longitude,
        a2.lat AS Dest_Latitude,
        a2.long AS Dest_Longitude
    FROM 
        "{year}" f 
    LEFT JOIN 
        "carriers" c ON f.UniqueCarrier = c.Code
    LEFT JOIN 
        "airports" a1 ON f.Origin = a1.iata 
    LEFT JOIN 
        "airports" a2 ON f.Dest = a2.iata   
    WHERE 
        f.Diverted IS NOT NULL;
    '''

    
    df_year = pd.read_sql_query(query, conn)
    df_year['Year'] = year 
    all_years_data.append(df_year)  


# In[127]:


# Combining all the yearly data into a single DataFrame
combined_data = pd.concat(all_years_data, ignore_index=True)

# Dropping missing values 
combined_data.dropna(inplace=True)



# In[145]:


# One-hot encoding for categorical variables
df_encoded_all = pd.get_dummies(combined_data, columns=['CarrierName', 'Month'], drop_first=True)

# Defining X and Y for logistic regression
X = df_encoded_all.drop('Diverted', axis=1)
y = df_encoded_all['Diverted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# In[146]:


# PData under-sampling

print("Data before undersampling:")

# Distribution of target variable before undersampling
print("Counts of data before undersampling:")
print(y_train.value_counts())

# Shape of the feature set before undersampling
print(f"Shape of X_train before undersampling: {X_train.shape}")
print(f"Shape of y_train before undersampling: {y_train.shape}")

# Random Under-sampling to balance the data
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Counts of the resampled data to check if the classes are balanced
print("\nCounts of resampled data after undersampling:")
print(y_resampled.value_counts())

# Shape of the resampled feature set
print(f"Shape of X_train after undersampling: {X_resampled.shape}")
print(f"Shape of y_train after undersampling: {y_resampled.shape}")


# In[147]:


# Logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# Predictions and probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for class 1

# Coefficients for the combined data
coefficients = model.coef_[0]
features = X.columns



# #### Analysis

# In[148]:


# Coefficient Table
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)



# In[151]:


coef_df.to_csv('coef_df.csv', index = False)


# In[152]:


conf_matrix


# In[153]:


report


# #### Visualization

# In[149]:


# ROC Curve and AUC Score
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Bar Plot for Coefficients
plt.figure(figsize=(10, 6))
coef_df.plot(kind='barh', x='Feature', y='Coefficient', legend=False)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()



# Histogram of Predicted Probabilities
plt.figure(figsize=(10, 6))
plt.hist(y_prob, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Predicted Probabilities")
plt.xlabel("Predicted Probability of Diversion")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[ ]:




