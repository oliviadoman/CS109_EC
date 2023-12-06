import pandas as pd

# Reading the eventing data into a pandas DataFrame
combined_df = pd.read_csv('fei_data.csv')

weights = {'Dressage_Penalties': 0.2, 'Cross_Country_Penalties': 0.5, 'Show_Jumping_Penalties': 0.3}


# Filtering the DataFrame for each athlete
athletes = ["Doug PAYNE", "Tamra SMITH", "Will COLEMAN"]  # Add other athlete names as needed
dataframes = {athlete: combined_df[combined_df['Athlete'].str.contains(athlete)] for athlete in athletes}

# Calculate average score for each athlete
for athlete, df in dataframes.items():
    df['average_score'] = df['Score'].mean()
    # Calculate Cross-Country penalties
    df['XC'] = df['XC Obs'] + df['XC Tim']
    # Calculate Show Jumping penalties
    df['SJ'] = df['J Obs'] + df['J Tim']

for athlete, df in dataframes.items():
    df['Composite_Score'] = (df['Dressage'] * weights['Dressage_Penalties'] +
                             df['XC'] * weights['Cross_Country_Penalties'] +
                             df['SJ'] * weights['Show_Jumping_Penalties'])

# MLE of the mean score and variance
mle_stats = {}
for athlete, df in dataframes.items():
    mean_mle = df['Score'].mean()
    variance_mle = df['Score'].var()
    mle_stats[athlete] = (mean_mle, variance_mle)

# Sort the athletes based on their mean MLE score in ascending order
sorted_performers = sorted(mle_stats, key=lambda x: mle_stats[x][0])

# Get the top three performers
top_three_performers = sorted_performers[:3]

# Print the top three performers
print("Top three performers based on MLE:")
for performer in top_three_performers:
    print(performer)
