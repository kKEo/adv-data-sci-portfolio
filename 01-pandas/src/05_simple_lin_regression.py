import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

data = pd.read_csv("../data/marketing_sales_data.csv")
print(data)

# Step 1. Start with .isna() to get booleans indicating whether each value in the data is missing

print(data.isna())
print(data.isna().any(axis=1))
print(data.isna().any(axis=1).sum())

data = data.dropna(axis=0)
print(data)

# Create plot of pairwise relationships
sns.pairplot(data)
# plt.show()

ols_data = data[["Radio", "Sales"]]
ols_formula = "Sales ~ Radio"
OLS = ols(formula=ols_formula, data=ols_data)
model = OLS.fit()
print(model.summary())

# Plot the OLS data with the best fit regression line
sns.regplot(x="Radio", y = "Sales", data=ols_data)

# Visualize the distribution of the residuals

residuals = model.resid
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()

# Create a Q-Q plot to confirm the assumption of normality.
sm.qqplot(residuals, line='s')
plt.title("Q-Q plot of Residuals")
plt.show()