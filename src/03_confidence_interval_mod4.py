import pandas as pd
import numpy as np

aqi = pd.read_csv('../data/c4_epa_air_quality.csv')


print(aqi.describe(include='all'))
print(aqi['state_name'].value_counts())

rre_states = ['California', 'Florida', 'Michigan', 'Ohio', 'Pennsylvania', 'Texas']

# Subset `aqi` to only consider these states.
aqi_rre = aqi[aqi['state_name'].isin(rre_states)]

# Find the mean aqi for each of the RRE states.
aqi_rre.groupby(['state_name']).agg({"aqi":"mean","state_name":"count"}) #alias as aqi_rre

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x=aqi_rre["state_name"],y=aqi_rre["aqi"])
# plt.show()

aqi_ca = aqi[aqi['state_name'] == 'California']
sample_mean = aqi_ca['aqi'].mean()


# Begin by identifying the z associated with your chosen confidence level.

z_value = 1.96

# Next, calculate your standard error.
standard_error = aqi_ca['aqi'].std() / np.sqrt(aqi_ca.shape[0])
print("standard error:")
print(standard_error)

# Lastly, use the preceding result to calculate your margin of error.
margin_of_error = standard_error * z_value
print("margin of error:")
print(margin_of_error)

upper_ci_limit = sample_mean + margin_of_error
lower_ci_limit = sample_mean - margin_of_error

# ----- OR ------

from scipy import stats
confidence_level = 0.05
stats.norm.interval(confidence=confidence_level, loc=sample_mean, scale=standard_error)
