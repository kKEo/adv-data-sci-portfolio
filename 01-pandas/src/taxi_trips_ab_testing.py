import pandas as pd
from scipy import stats

taxi_data = pd.read_csv("../data/text-trip-data.csv", index_col=0)

pd.set_option('display.max_columns', None)
print(taxi_data.describe(include='all'))

print("-"*41)
print("-- Mean of Fare_Amount grouped by Payment_Type --")
print(taxi_data.groupby('payment_type')['fare_amount'].mean())

# H0: There is no difference in average fare between
# customers who use credit cards and customers who use cash.

# Ha: There is a difference in average fare
# between customers who use credit cards and customers who use cash

credit_card = taxi_data[taxi_data['payment_type'] == 1]['fare_amount']
cash = taxi_data[taxi_data['payment_type'] == 2]['fare_amount']

ttest_result = stats.ttest_ind(a=credit_card, b=cash, equal_var=False)
print(ttest_result)

if ttest_result.pvalue < 0.05:
    print("Null hypothesis rejected!!")
    print("As a conclusion we can state there is a statistically "
          "significant difference in the average fare amount between"
          "customers paying with card and who use cash")
else:
    print("Failed to reject null hypothesis")

