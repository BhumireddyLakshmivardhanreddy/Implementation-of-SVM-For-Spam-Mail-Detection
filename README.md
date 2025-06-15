# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BHUMIREDDY LAKSHMI VARDHAN REDDY
RegisterNumber:  212223240016
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()


data.info()


data.isnull().sum()


x=data["v2"].values
y=data["v1"].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()


x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)

```
## Output:
![image](https://github.com/user-attachments/assets/79b3860f-56fc-4021-b74e-3a46b27eb2c1)
![image](https://github.com/user-attachments/assets/b1c6e0f4-ea37-4123-bc7a-62646f7f4fec)
![image](https://github.com/user-attachments/assets/63b85d1b-5dc8-4981-b2de-165984ffdc04)
![image](https://github.com/user-attachments/assets/8930e3e0-515e-4640-b368-0f6e35f22a57)
![image](https://github.com/user-attachments/assets/cd94d6e9-2096-434c-a0c8-ec44fce913e1)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
