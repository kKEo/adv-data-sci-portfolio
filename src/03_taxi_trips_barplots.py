

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv('../data/text-trip-data.csv')

# print(df.size)
# df.info()

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])


def plot(dataframe):
    plt.figure(figsize=(7, 2))
    plt.title('trip_distance')
    sns.boxplot(data=None, x=df['trip_distance'], fliersize=1)
    plt.show()


# plot(df)


def histogram():
    plt.figure(figsize=(10, 5))
    sns.histplot(df['trip_distance'], bins=range(0, 26, 1))
    plt.title('Trip distance histogram')
    plt.show()

# histogram()


def total_amount_boxplot():
    plt.figure(figsize=(7, 2))
    plt.title('total_amount')
    sns.boxplot(x=df['total_amount'], fliersize=1)
    plt.show()


# total_amount_boxplot()


def total_amount_histogram():
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['total_amount'], bins=range(-10, 101, 5))
    ax.set_xticks(range(-10, 101, 5))
    ax.set_xticklabels(range(-10, 101, 5))
    plt.title('Total amount histogram')
    plt.show()


# total_amount_histogram()

def tip_amount_by_vendor():
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(data=df, x='tip_amount', bins=range(0, 21, 1),
                      hue='VendorID',
                      multiple='stack',
                      palette='pastel')
    ax.set_xticks(range(0, 21, 1))
    ax.set_xticklabels(range(0, 21, 1))
    plt.title('Tip amount by vendor histogram')


def tip_amount_over_10_by_vendoer():
    over_10_filter = df['tip_amount'] > 10
    tips_over_ten = df[over_10_filter]
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(data=tips_over_ten, x='tip_amount', bins=range(10, 21, 1),
                      hue='VendorID',
                      multiple='stack',
                      palette='pastel')
    ax.set_xticks(range(10, 21, 1))
    ax.set_xticklabels(range(10, 21, 1))
    plt.title('Tip amount by vendor histogram')
    plt.show()


def passenger_counts():
    df['passenger_count'].value_counts()


def mean_tip_passenger_counts():
    mean_tips_by_passenger_count = df.groupby(['passenger_count']).mean()[['tip_amount']]
    data = mean_tips_by_passenger_count.tail(-1)
    pal = sns.color_palette("Greens_d", len(data))
    rank = data['tip_amount'].argsort().argsort()
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=data.index,
                     y=data['tip_amount'],
                     palette=np.array(pal[::-1])[rank])
    ax.axhline(df['tip_amount'].mean(), ls='--', color='red', label='global mean')
    ax.legend()
    plt.title('Mean tip amount by passenger count', fontsize=16)


def total_ride_count_per_month():
    df['month'] = df['tpep_pickup_datetime'].dt.month_name()
    df['day'] = df['tpep_pickup_datetime'].dt.day_name()
    monthly_rides = df['month'].value_counts()

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']

    monthly_rides = monthly_rides.reindex(index=month_order)

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=monthly_rides.index, y=monthly_rides)
    ax.set_xticklabels(month_order)
    plt.title('Ride count by month', fontsize=16)
    plt.show()


def total_rides_by_day_of_a_week():
    daily_rides = df['day'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_rides = daily_rides.reindex(index=day_order)
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=daily_rides.index, y=daily_rides)
    ax.set_xticklabels(day_order)
    ax.set_ylabel('Count')
    plt.title('Ride count by day', fontsize=16)
    plt.show()


def distance():
    # Calculate the mean trip distance for each drop-off location
    distance_by_dropoff = df.groupby('DOLocationID').mean([['trip_distance']])
    # Sort the results in descending order by mean trip distance
    distance_by_dropoff = distance_by_dropoff.sort_values(by='trip_distance')

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=distance_by_dropoff.index,
                     y=distance_by_dropoff['trip_distance'],
                     order=distance_by_dropoff.index)
    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.title('Mean trip distance by drop-off location', fontsize=16)
    plt.show()


distance()
