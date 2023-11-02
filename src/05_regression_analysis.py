import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the statsmodel module.
import statsmodels.api as sm

# Import the ols function from statsmodels.
from statsmodels.formula.api import ols

data = pd.read_csv('../data/marketing_and_sales_data_evaluate_lr.csv')

print(data[['TV', 'Radio', 'Social_Media']].describe())

# Calculate the average missing rate in the sales column.
missing_sales = data.Sales.isna().mean()

missing_sales = round(missing_sales*100, 2)
print(f'Percentage of promotions missing Sales: {missing_sales}%')

data = data.dropna(subset=['Sales'], axis = 0)

fig = sns.histplot(data['Sales'])
fig.set_title('Distribution of Sales')

# sns.pairplot(data)
# plt.show()


# Define the OLS formula.
ols_formula = 'Sales ~ TV'

# Create an OLS model.
OLS = ols(formula = ols_formula, data = data)

# Fit the model.
model = OLS.fit()

# Save the results summary.
model_results = model.summary()

print(model_results)

plt.close()

# Model Assumptions

# Linearity - Create a scatterplot comparing X and Sales (Y).
sns.scatterplot(x=data['TV'], y=data['Sales'])
# plt.show()

# Normality - Calculate the residuals.

residuals = model.resid

# Create a 1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Create a histogram with the residuals .
sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")

# Set the title of the residual plot.
axes[0].set_title("Histogram of Residuals")

# Create a Q-Q plot of the residuals.
sm.qqplot(residuals, line='s', ax=axes[1])

# Set the title of the Q-Q plot.
axes[1].set_title("Normal Q-Q plot")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()

# Show the plot.
# plt.show()
plt.close()

# Homoscedasticity - Create a scatterplot with the fitted values from the model and the residuals.

fig = sns.scatterplot(x=model.fittedvalues, y=model.resid)

fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)
# plt.show()


print(model_results)
