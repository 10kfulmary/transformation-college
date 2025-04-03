import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#load the data set
data = pd.read_csv("Student_Performance.csv")

print (data.head()) #print the first five rows of the data set
print ("column in dataset:", data.columns)

# Convert categorical values in "Extracurricular Activities" to numeric (Yes=1, No=0)
data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes": 1, "No": 0})

#select features ans target variables
X = data [["Hours Studied", "Previous Scores", "Extracurricular Activities"]]
y = data ["Performance Index"] #target variable

#print the shape of the dataset
print("\nFeatures Shape:", X.shape)
print ("Target Shape:", y.shape)

#display first five rows
print(X.head())

#split into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print ("Model training complete!")

#make pedictions please note the word model.predict
y_pred = model.predict (X_test)

#print some predicted values
print ("Predicted Performance Index:", y_pred[:5])

#calculate error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error (y_test, y_pred)
r2 = r2_score (y_test, y_pred)

print (f"Mean Aboslute Error: {mae}")
print (f"Mean Squared Error: {mse}")
print (f"r2: {r2}")
