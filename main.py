# Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# importing the dataset
df=pd.read_csv('boston_house_data.csv')
house_price_dataframe=df.rename(columns={'MEDV':'price'})
# print(house_price_dataframe.head())
# print(house_price_dataframe.shape)
# print(house_price_dataframe.isnull().sum())

# statistical measures of the dataframe
# print(house_price_dataframe.describe())

# finding corelation between two variables
correlation=house_price_dataframe.corr()

# construct a heatmap
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
# plt.show()

# split data and target
X=house_price_dataframe.drop(['price'],axis=1)
y=house_price_dataframe['price']

# split into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

# model training
model=XGBRegressor()
model.fit(X_train,y_train)

# Prediction on training data
training_data_prediction=model.predict(X_train)
# print(training_data_prediction)

# R squared error
score_1=metrics.r2_score(y_train, training_data_prediction)
# mean absolute error
score_2=metrics.mean_absolute_error(y_train,training_data_prediction)
print("R squared error: ",score_1)
print("Mean absolute error: ",score_2)

# visualizing the actual and predicted price
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

# prediction on test data
test_data_prediction=model.predict(X_test)
# R squared error
score_1=metrics.r2_score(y_test, test_data_prediction)
# mean absolute error
score_2=metrics.mean_absolute_error(y_test,test_data_prediction)
print("R squared error: ",score_1)
print("Mean absolute error: ",score_2)