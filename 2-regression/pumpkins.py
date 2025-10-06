import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime

# https://youtu.be/5qGjczWTrDQ?list=PLlrxD0HtieHjNnGcZ1TWzPjKYWgfXSiWG

# Read the US pumpkins CSV file
pumpkins = pd.read_csv('US-pumpkins.csv')

# Filter out any rows that don't use price per bushel
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
print(pumpkins.head())

# shows empty cells per columns
print(pumpkins.isnull().sum())

# Define the colums we want to keep
new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']

# Drop all other columns
pumpkins = pumpkins.drop([ c for c in pumpkins.columns if c not in new_columns], axis=1)
# same as pumpkins = pumpkins.loc[:, columns_to_select]

# Calculate the average price from the high and low price columns
price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

# Get the month from the date column 
month = pd.DatetimeIndex(pumpkins['Date']).month
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)

# Create a new data frame with these columns
new_pumpkins = pd.DataFrame({
    'Month': month,
    'DayOfYear': day_of_year,
    'Package': pumpkins['Package'],
    'Low Price': pumpkins['Low Price'],
    'High Price': pumpkins['High Price'],
    'Price': price
})

# Convert the price of all cells price by 1 1/9 bushels by diving by 1 1/9
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1+ 1/9);

# Convert the price of all cells prices by 1/2 bushels by dividing by 1/2
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)

# print(new_pumpkins.tail())

# Get the values we want to plot
price = new_pumpkins.Price
month = new_pumpkins.Month

# Create and show a scatter plot of price vs month
# plt.scatter(price, month)
# plt.show()
# This graph isn't the best to show the data

new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
plt.ylabel('Pumpkin Price')
plt.show();
