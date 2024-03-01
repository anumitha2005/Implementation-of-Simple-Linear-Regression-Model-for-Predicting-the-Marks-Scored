# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import necessary libraries (e.g., pandas, numpy,matplotlib).
2.Load the dataset and then split the dataset into training and testing sets using sklearn library.
3.Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4.Use the trained model to predict marks based on study hours in the test dataset.
5.Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ANUMITHA.M.R
RegisterNumber: 212223040018 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```







## Output:
df.head()
![image](https://github.com/anumitha2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155522855/bf3c561d-5f79-4824-be1b-7f38a164519c)
Array value of x:

![Screenshot 2024-03-01 055157](https://github.com/anumitha2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155522855/8be6e112-b983-4e48-80b5-9dcf253de6d2)
Array value of y:

![image](https://github.com/anumitha2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155522855/eb3d4cdf-3a0b-44a1-a228-be30361d18b8)
values of predicition:

![image](https://github.com/anumitha2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155522855/317d8f9a-5d07-4f6a-bb63-f7cd98a60841)
taning set:

![image](https://github.com/anumitha2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155522855/f32cb211-187e-4abf-acfc-07cee6d1a2ef)
value of MSE,MAE AND RMSE:

![image](https://github.com/anumitha2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155522855/2c90b536-d4e4-42ae-8c18-7778ab674084)












## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
