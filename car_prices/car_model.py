from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("car_prices.csv")
print (data.head()) #output the top five rows in the car_prices.csv file above
#print(data.columns)
data = data.dropna() #to drop all missing rows
#data.info()  #to check the data types of all the columns
#data.isna().sum() #sumtotal of rows not available

label_encoder = LabelEncoder() #encoding of all datatypes
data['Mileage'] = label_encoder.fit_transform(data['Mileage'])
data['fuel type'] = label_encoder.fit_transform(data['fuel type'])
data['Model'] = label_encoder.fit_transform(data['Model'])


features = ['Year of manufacture','Mileage','Engine Size','Model','fuel type']
X = data[features]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print("model training complete")

predictions = model.predict(X_test)
print(predictions[:5])

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(mae)
print(rmse)
