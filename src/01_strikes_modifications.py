import pandas as pd
import seaborn as sns


df = pd.read_csv('eda_structuring_with_python_dataset1.csv')
df['date'] = pd.to_datetime(df['date'])
df.shape

df.drop_duplicates().shape


df.sort_values(by='number_of_strikes', ascending=False).head(10)

df.center_point_geom.value_counts()
df['week'] = df.date.dt.isocalendar().week
df['weekday'] = df.date.dt.day_name()

df[['weekday', 'number_of_strikes']].groupby(['weekday']).mean()

# Identify the top 20 locations with most days of lightning.
df.center_point_geom.value_counts()[:20].rename_axis('unique_values').reset_index(name='counts').style.background_gradient()


# Create boxplots of strike counts for each day of week.
g = sns.boxplot(data=df,
                x='weekday',
                y='number_of_strikes',
                order=weekday_order,
                showfliers=False
                )
g.set_title('Lightning distribution per weekday (2018)')

