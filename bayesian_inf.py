import pandas as pd
import numpy as np
import scipy as stats

# Read data
eventing_df = pd.read_csv('eventing_data_2.csv')

# Assume a prior normal distribution for scores (mean=35, std=6)
prior_mean = 35
prior_std = 6

athletes = ["Doug PAYNE", "Tamra SMITH", "William COLEMAN", "Phillip DUTTON", "Elisabeth HALLIDAY", "James ALLISTON" ]  # Add other athlete names as needed
dataframes = {athlete: eventing_df[eventing_df['Athlete'].str.contains(athlete)] for athlete in athletes}
weights = {'Dressage_Penalties': 0.25, 'Cross_Country_Penalties': 0.25, 'Show_Jumping_Penalties': 0.25, 'Overall_Placing': 0.25} # Adjusted weights to include Overall Placing

athlete_posteriors = {}

for athlete in athletes:
    athlete_data = eventing_df[eventing_df['Athlete'].str.contains(athlete)]
    athlete_data['Composite_Score'] = (athlete_data['Dressage'] * weights['Dressage_Penalties'] +
                                       athlete_data['XC obs'] + athlete_data['XC tim'] * weights['Cross_Country_Penalties'] +
                                       athlete_data['SJ obs'] + athlete_data['SJ tim'] * weights['Show_Jumping_Penalties'] +
                                       athlete_data['Position'] * weights['Overall_Placing']) # Included Overall Placing in the composite score

    # Compute the sample mean and standard deviation for the composite score
    sample_mean = athlete_data['Composite_Score'].mean()
    sample_size = len(athlete_data)
    sample_std = athlete_data['Composite_Score'].std()
    sample_variance = athlete_data['Composite_Score'].var(ddof=1)


    # Calculate posterior distribution parameters
    posterior_mean = (prior_mean/prior_std**2 + sample_mean*sample_size/sample_std**2) / (1/prior_std**2 + sample_size/sample_std**2)
    posterior_std = np.sqrt(1 / (1/prior_std**2 + sample_size/sample_std**2))

    # Store the posterior mean
    athlete_posteriors[athlete] = (posterior_mean, posterior_std)

# Find the athlete with the best (lowest) posterior mean
sorted_athletes = sorted(athlete_posteriors, key=athlete_posteriors.get)

# Print the top three performers
print("Top four performers based on Bayesian Inference:")
for idx, performer in enumerate(sorted_athletes[:4], 1):
    posterior_mean, posterior_std = athlete_posteriors[performer]
    print(f"\n{idx}. Athlete: {performer}")
    print(f"   Posterior Mean: {posterior_mean}")
    print(f"   Posterior Standard Deviation: {posterior_std}")