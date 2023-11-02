import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


df = pd.read_csv('eda_using_basic_data_functions_in_python_dataset1.csv')

df.head(10)

df['date']= pd.to_datetime(df['date'])


# count strikes per date
df.groupby(['date']).sum().sort_values('number_of_strikes', ascending=False).head(10)

# extract just month, add as separate column
df['month'] = df['date'].dt.month
# df.head()

# count strikes per month
df.groupby(['month']).sum().sort_values('number_of_strikes', ascending=False).head(12)


#
df_by_month = df.groupby(['month','month_txt']).sum().sort_values('month', ascending=True).head(12).reset_index()

# draw a plot

plt.bar(x=df_by_month['month_txt'], height=df_by_month['number_of_strikes'], label="Number of strikes")
plt.plot()

plt.xlabel("Months(2018)")
plt.ylabel("Number of lightning strikes")
plt.title("Number of lightning strikes in 2018 by months")
plt.legend()
plt.show()
