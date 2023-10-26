import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols

diamonds = sns.load_dataset("diamonds")
# Subset for color, cut, price columns
diamonds2 = diamonds[["color", "cut", "price"]]

# Only include colorless diamonds
diamonds2 = diamonds2[diamonds2["color"].isin(["E","F","H","D","I"])]

# Drop removed colors, G and J
diamonds2.color = diamonds2.color.cat.remove_categories(["G","J"])

# Only include ideal, premium, and very good diamonds
diamonds2 = diamonds2[diamonds2["cut"].isin(["Ideal","Premium","Very Good"])]

# Drop removed cuts
diamonds2.cut = diamonds2.cut.cat.remove_categories(["Good","Fair"])

# Drop NaNs
diamonds2.dropna(inplace = True)

# Reset index
diamonds2.reset_index(inplace = True, drop = True)

# Add column for logarithm of price
diamonds2.insert(3, "log_price", [math.log(price) for price in diamonds2["price"]])

model2 = ols(formula="log_price ~ C(color) + C(cut) + C(color):C(cut)", data=diamonds2).fit()

print(model2.summary())

#  𝐻0:𝑝𝑟𝑖𝑐𝑒𝐷=𝑝𝑟𝑖𝑐𝑒𝐸=𝑝𝑟𝑖𝑐𝑒𝐹=𝑝𝑟𝑖𝑐𝑒𝐻=𝑝𝑟𝑖𝑐𝑒𝐼
# 𝐻1:Not 𝑝𝑟𝑖𝑐𝑒𝐷=𝑝𝑟𝑖𝑐𝑒𝐸=𝑝𝑟𝑖𝑐𝑒𝐹=𝑝𝑟𝑖𝑐𝑒𝐻=𝑝𝑟𝑖𝑐𝑒𝐼

print(sm.stats.anova_lm(model2, typ=2))
print(sm.stats.anova_lm(model2, typ=1))
print(sm.stats.anova_lm(model2, typ=3))


# Import Tukey's HSD function
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Run Tukey's HSD post hoc test for one-way ANOVA
tukey_oneway = pairwise_tukeyhsd(endog=diamonds2["log_price"], groups=diamonds2["color"], alpha = 0.05)

# Get results (pairwise comparisons)
print(tukey_oneway.summary())

