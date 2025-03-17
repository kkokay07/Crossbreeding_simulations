import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Population size
n_individuals = 1000

# Updated means and standard deviations for traits
egg_means = {
    "Leghorn": 330,  # Egg production at 72 weeks
    "Aseel": 60,
    "Conventional": 180,
    "CCB (Chr 13)": 250
}

bw_means = {
    "Leghorn": 1.3,  # Body weight at 20 weeks
    "Aseel": 1.85,
    "Conventional": 1.6,
    "CCB (Chr 13)": 1.9
}

egg_sds = {
    "Leghorn": 20,
    "Aseel": 10,
    "Conventional": 15,
    "CCB (Chr 13)": 20
}

bw_sds = {
    "Leghorn": 0.2,
    "Aseel": 0.3,
    "Conventional": 0.3,
    "CCB (Chr 13)": 0.3
}

# Simulating Egg Production and Body Weight
egg_data = {"Breeding": [], "Egg Production": []}
bw_data = {"Breeding": [], "Body Weight (kg)": []}

for breed in egg_means.keys():
    egg_data["Breeding"].extend([breed] * n_individuals)
    egg_data["Egg Production"].extend(np.random.normal(egg_means[breed], egg_sds[breed], n_individuals))
    
    bw_data["Breeding"].extend([breed] * n_individuals)
    bw_data["Body Weight (kg)"].extend(np.random.normal(bw_means[breed], bw_sds[breed], n_individuals))

# Converting to DataFrames
df_egg = pd.DataFrame(egg_data)
df_bw = pd.DataFrame(bw_data)

# Plotting Egg Production (72 weeks) and Body Weight (20 weeks)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Egg Production Boxplot (at 72 weeks)
sns.boxplot(data=df_egg, x='Breeding', y='Egg Production', 
            palette=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'], ax=axes[0])
axes[0].set_title('Egg Production at 72 Weeks', fontsize=16)
axes[0].set_xlabel('Breeding Strategy', fontsize=14)
axes[0].set_ylabel('Eggs Laid', fontsize=14)
axes[0].tick_params(axis='x', rotation=15)

# Body Weight Boxplot (at 20 weeks)
sns.boxplot(data=df_bw, x='Breeding', y='Body Weight (kg)', 
            palette=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'], ax=axes[1])
axes[1].set_title('Body Weight at 20 Weeks', fontsize=16)
axes[1].set_xlabel('Breeding Strategy', fontsize=14)
axes[1].set_ylabel('Body Weight (kg)', fontsize=14)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# Save results
df_egg.to_csv("egg_production_72weeks.csv", index=False)
df_bw.to_csv("body_weight_20weeks.csv", index=False)

print("Simulation completed. Data saved.")
