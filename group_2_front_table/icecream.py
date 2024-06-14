import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Read the data from local repo
data = pd.read_csv('/Users/kristianlarsen/Desktop/Reprod_course.csv', sep=',')

# Convert columns to numeric, forcing errors to NaN and dropping them
data['temperature..Celsius.'] = pd.to_numeric(data['temperature..Celsius.'], errors='coerce')
data['ice.cream....scoops.'] = pd.to_numeric(data['ice.cream....scoops.'], errors='coerce')
data.dropna(subset=['temperature..Celsius.', 'ice.cream....scoops.'], inplace=True)

# Remove data points where temperature or scoops are less than 0
data = data[(data['temperature..Celsius.'] >= 0) & (data['ice.cream....scoops.'] >= 0)]

# Prepare data for linear regression
X = data[['temperature..Celsius.']]
y = data['ice.cream....scoops.']

# Fit the linear model with intercept forced through zero
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

# Predict the number of scoops at 40 degrees Celsius
predicted_scoops = model.predict(np.array([[40]]))

# Print the predicted scoops
print(f"The predicted number of ice cream scoops at 40 degrees Celsius is: {predicted_scoops[0]}")

# Create a new data frame for the regression line
temperature_range = np.linspace(data['temperature..Celsius.'].min(), data['temperature..Celsius.'].max(), 100)
regression_line = pd.DataFrame({
    'temperature..Celsius.': temperature_range,
    'ice.cream....scoops.': model.predict(temperature_range.reshape(-1, 1))
})

# Plot the data with the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temperature..Celsius.', y='ice.cream....scoops.', data=data, color='blue')
sns.lineplot(x='temperature..Celsius.', y='ice.cream....scoops.', data=regression_line, color='red')
plt.title('Ice Cream Scoops vs Temperature (Intercept Forced Through Zero)')
plt.xlabel('Temperature (Celsius)')
plt.ylabel('Ice Cream Scoops')
plt.show()
