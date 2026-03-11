# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load employee data and split it into training and testing sets.

2.Train a Decision Tree classifier using entropy as the split criterion.

3.Evaluate the model using accuracy, confusion matrix, and classification report.

4.Use the trained model to predict whether a new employee will stay or leave.
```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rohith G
RegisterNumber:  212225040347
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy",random_state=100)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
sample=[[0.5,0.8,9,260,6,0,1,2]]
print("Prediction for sample:",dt.predict(sample))
plt.figure(figsize=(12,8))
plot_tree(dt,feature_names=x.columns,class_names=["stayed","left"],filled=True,rounded=True,fontsize=10)
plt.show()
```

## Output:
<img width="1044" height="349" alt="Screenshot 2026-02-26 104108" src="https://github.com/user-attachments/assets/5bdbb1d7-fb4f-45b2-9903-f0cbdfed9ec3" />
<img width="1515" height="522" alt="Screenshot 2026-02-26 105257" src="https://github.com/user-attachments/assets/869678be-351e-406a-9cee-ebc00a3409cf" />
<img width="1378" height="721" alt="Screenshot 2026-02-26 105317" src="https://github.com/user-attachments/assets/6d29b0e0-2b9e-4064-9642-40374da332c9" />
<img width="1399" height="516" alt="Screenshot 2026-02-26 105334" src="https://github.com/user-attachments/assets/c5642624-0d34-442a-ab50-e1d32027edcd" />
<img width="1249" height="741" alt="Screenshot 2026-02-26 105356" src="https://github.com/user-attachments/assets/fa98c315-991e-44bd-80e6-f31f8ad0b037" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
