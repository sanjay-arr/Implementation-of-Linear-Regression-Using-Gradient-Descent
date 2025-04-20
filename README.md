# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.
2.write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using gradient descent and generate the required graph
```

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: G SANJAY 
RegisterNumber:  212224230243
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    #Add a column of ones to x for the intercept term
    x=np.c_[np.ones(len(x1)),x1]
    #Initialize thetha with zeros
    thetha=np.zeros(x.shape[1]).reshape(-1,1)
    #Perform gradient descent
    for i in range(num_iters):
        #Calculate predictions
        predictions=(x).dot(thetha).reshape(-1,1)
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update thetha using gradient descent
        thetha-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return thetha
data=pd.read_csv("C:\\Users\\admin\\Downloads\\50_Startups.csv",header=None)
#Assuming the last column is your target variable 'y' and the preceding columns are your features 'x'
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
#Learn model parameters
thetha=linear_regression(x1_scaled,y1_scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),thetha)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")  
```

## Output:
![image](https://github.com/user-attachments/assets/f1ac0346-52ff-4753-9731-b0ea4c5fa1c7)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
