# isso é a versão que o chatgpt deu de sugestão

import pandas as pd
import numpy as np

# Load dataset from CSV file
df = pd.read_csv('data/timeseries_NEW.csv')

print(df.head(5))
# Reshape the dataset to have one column for time series and one column for individual ID
df = df.melt(id_vars = 0 ,var_name='minute', value_name='activity')

print(df.head(5))

# Add a new column for timestamp
df['timestamp'] = pd.date_range(start='1/1/2023', periods=len(df), freq='T')

print(df.head(5))

# Create a new dataframe to store the predictors
predictors_df = pd.DataFrame()

# Add predictors to the new dataframe
predictors_df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.time
predictors_df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
predictors_df['rolling_mean_5min'] = df.groupby('individual')['activity'].rolling(5).mean().values
predictors_df['rolling_mean_15min'] = df.groupby('individual')['activity'].rolling(15).mean().values
predictors_df['rolling_mean_1hour'] = df.groupby('individual')['activity'].rolling(60).mean().values
predictors_df['fourier_transform'] = np.fft.fft(df['activity'].values).real
predictors_df['mean'] = df.groupby('individual')['activity'].mean().values
predictors_df['std'] = df.groupby('individual')['activity'].std().values
predictors_df['skewness'] = df.groupby('individual')['activity'].skew().values
predictors_df['kurtosis'] = df.groupby('individual')['activity'].kurt().values

# Print the first few rows of the new dataframe
print(predictors_df.head())
