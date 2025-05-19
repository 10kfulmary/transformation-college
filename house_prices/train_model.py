import pandas as pd # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

data = pd.read_csv("train.csv")  # Load dataset

#print(data.columns)  # Check actual column names

X = data[["BedroomAbvGr", "GrLivArea", "OverallQual"]]  # feature names
y = data["SalePrice"]  #target variable

# Print the shape of the dataset
#print("\nFeatures Shape:", X.shape)
#print("Target Shape:", y.shape)

# Display first few rows
#print(X.head())

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#print("Training set size:", X_train.shape)
#print("Test set size:", X_test.shape)

#train the model
model = LinearRegression()
model.fit(X_train, y_train)
# print ("Model training complete!")

# Make predictions please note the word model.predict
y_pred = model.predict(X_test)

# Print some predicted values
# print("Predicted prices:", y_pred[:5])

# Calculate error
error = mean_absolute_error(y_test, y_pred)
print (f"Mean Absolute Error: {error}")

# Visualize the data

# 1. Bar charts of features vs. target variable
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.barplot(x=X["BedroomAbvGr"], y=y)
plt.title("BedroomAbvGr vs. SalePrice (Bar Chart)")
plt.subplot(1, 3, 2)
sns.barplot(x=X["GrLivArea"], y=y)
plt.title("GrLivArea vs. SalePrice (Bar Chart)")
plt.subplot(1, 3, 3)
sns.barplot(x=X["OverallQual"], y=y)
plt.title("OverallQual vs. SalePrice (Bar Chart)")
plt.tight_layout()
plt.show()

# 2. KDE plots of features
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(X["BedroomAbvGr"], kde=True)
plt.title("BedroomAbvGr Distribution (KDE)")
plt.subplot(1, 3, 2)
sns.histplot(X["GrLivArea"], kde=True)
plt.title("GrLivArea Distribution (KDE)")
plt.subplot(1, 3, 3)
sns.histplot(X["OverallQual"], kde=True)
plt.title("OverallQual Distribution (KDE)")
plt.tight_layout()
plt.show()

# 3. Predicted vs. Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.show()
