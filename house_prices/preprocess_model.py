import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("train.csv")  # Load dataset

#print(data.columns)  # Check actual column names

X = data[["BedroomAbvGr", "GrLivArea", "OverallQual"]]  # feature names
y = data["SalePrice"]  #target variable

# Print the shape of the dataset
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Display first few rows
print(X.head())

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Training set size:", X_train.shape)
print("Training set size:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print ("Model training complete!")

# Make predictions please note the word model.predict
y_pred = model.predict(X_test)

# Print some predicted values
print("Predicted prices:", y_pred[:5])

# Calculate error
error = mean_absolute_error(y_test, y_pred)
print (f"Mean Absolute Error: {error}")