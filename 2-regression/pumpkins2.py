import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 

# https://youtu.be/5qGjczWTrDQ?list=PLlrxD0HtieHjNnGcZ1TWzPjKYWgfXSiWG

# Read the US pumpkins CSV file
pumpkins = pd.read_csv('US-pumpkins.csv')

# Filter out any rows that don't use price per bushel
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
print(pumpkins.head())

# shows empty cells per columns
print(pumpkins.isnull().sum())

# Define the colums we want to keep
new_columns = ['Package', 'Month', 'Variety', 'Low Price', 'High Price', 'Date']

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
    'Variety': pumpkins['Variety'], 
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

# new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
# plt.ylabel('Pumpkin Price')
# plt.show();

# plt.scatter('DayOfYear', 'Price', data=new_pumpkins)
# plt.show()

print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))

# Define the colors to use to plot the pumpkins
colors = ['red', 'blue', 'green', 'yellow']

# Plot the price vs day of year for the pumpkins, using a different color for each variety
ax=None
for i, var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety'] == var]
    ax = df.plot.scatter('DayOfYear', 'Price', ax=ax, c=colors[i], label=var)

# plt.show()
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')

# plt.show()
# PIE TYPE = type of pumpkins
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear', 'Price')
# plt.show()

print(pie_pumpkins['Month'].corr(pie_pumpkins['Price']))
print(pie_pumpkins['DayOfYear'].corr(pie_pumpkins['Price']))

# Get the day of year and price in separate arrays
# needs to reshape the X variable so it can be used
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1, 1)
y = pie_pumpkins['Price']

print(X.shape)

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

# Create a linear regression object
lin_reg = LinearRegression()

# Train the model using our training data
lin_reg.fit(X_train, y_train)

# Test the model using our test data
pred = lin_reg.predict(X_test)

# Calculate the mean squared error
mse = np.sqrt(mean_squared_error(y_test, pred))

# Print the mean squared error in an easy to read format
print(f'Mean Error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
# this gives a big error so predictions wont be great

# Calculate the coefficient of determination
score = lin_reg.score(X_train, y_train)
print('Model determination', score)

# Create a scatter plot using out test data
plt.scatter(X_test, y_test)
plt.plot(X_test, pred)
