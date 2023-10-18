import pandas as pd

df = pd.read_csv("../data/text-trip-data.csv")
df_sort = df.sort_values(by=['trip_distance'], ascending=False)
res = df_sort.head(10)

# Sort the data by total amount and print the top 20 values
total_amount_sorted = df.sort_values(['total_amount'], ascending=False)['total_amount']
res = total_amount_sorted.head(20)

# Sort the data by total amount and print the bottom 20 values
res = total_amount_sorted.tail(20)

# How many of each payment type are represented in the data?
res = df['payment_type'].value_counts()

# What is the average tip for trips paid for with credit card?
avg_cc_tip = df[df['payment_type']==1]['tip_amount'].mean()
print('Avg. cc tip:', avg_cc_tip)

# What is the average tip for trips paid for with cash?
avg_cash_tip = df[df['payment_type']==2]['tip_amount'].mean()
print('Avg. cash tip:', avg_cash_tip)


# What is the mean total amount for each vendor?
df.groupby(['VendorID']).mean(numeric_only=True)[['total_amount']]


# Filter the data for credit card payments only
credit_card = df[df['payment_type']==1]

# Filter the credit-card-only data for passenger count only
credit_card['passenger_count'].value_counts()

# Calculate the average tip amount for each passenger count (credit card payments only)
credit_card.groupby(['passenger_count']).mean(numeric_only=True)[['tip_amount']]

print(res)