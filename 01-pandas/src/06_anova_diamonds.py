import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

diamonds = sns.load_dataset("diamonds")
print(diamonds.head())

# Check how many diamonds are each color grade
print(diamonds["color"].value_counts())

# Subset for colorless diamonds
colorless = diamonds[diamonds["color"].isin(["E","F","H","D","I"])]

# Select only color and price columns, and reset index
colorless = colorless[["color","price"]].reset_index(drop=True)

# Remove dropped categories of diamond color
colorless.color = colorless.color.cat.remove_categories(["G","J"])

# Check that the dropped categories have been removed
colorless["color"].values

import math

# Take the logarithm of the price, and insert it as the third column
colorless.insert(2, "log_price", [math.log(price) for price in colorless["price"]])

# Drop rows with missing values
colorless.dropna(inplace=True)
# Reset index
colorless.reset_index(inplace=True, drop=True)

# Save to diamonds.csv
colorless.to_csv('../output/diamonds.csv', index=False, header=list(colorless.columns))

diamonds = colorless
sns.boxplot(x="color", y="log_price", data=diamonds)
# plt.show()

import statsmodels.api as sm
from statsmodels.formula.api import ols


# Construct simple linear regression model, and fit the model
model = ols(formula="log_price ~ C(color)", data=diamonds).fit()
print(model.summary())


# Run one-way ANOVA
print("="*21)
print("One-way Anova, Typ 2")
print(sm.stats.anova_lm(model, typ=2))
print("="*21)
print("One-way Anova, Typ 2")
sm.stats.anova_lm(model, typ=1)
print("="*21)
print("One-way Anova, Typ 2")
print(sm.stats.anova_lm(model, typ=3))

