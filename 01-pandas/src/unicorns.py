import pandas as pd
import matplotlib.pyplot as plt

companies = pd.read_csv("../data/Unicorn_Companies.csv")

companies["Date Joined"] = pd.to_datetime(companies["Date Joined"])
companies["Year Joined"] = companies["Date Joined"].dt.year

companies_sample = companies.sample(n=50, random_state=42)

companies_sample["Years Till Unicorn"] = companies_sample["Year Joined"] - companies_sample["Year Founded"]

by_industry = (companies_sample[["Industry", "Years Till Unicorn"]]
               .groupby("Industry")
               .max()
               .sort_values(by="Years Till Unicorn")
               )

plt.bar(by_industry.index, by_industry["Years Till Unicorn"])
plt.title("Bar plot of maximum years taken by company to become unicorn per industry (from sample)")
plt.xlabel("Industry")
plt.ylabel("Maximum number of years")
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()

valuation_billions = 'Valuation Billions'
companies_sample[valuation_billions] = companies_sample['Valuation']
# Remove the '$' from each value
companies_sample[valuation_billions] = companies_sample[valuation_billions].str.replace('$', '')
# Remove the 'B' from each value
companies_sample[valuation_billions] = companies_sample[valuation_billions].str.replace('B', '')
# Convert column to type int
companies_sample[valuation_billions] = companies_sample[valuation_billions].astype('int')