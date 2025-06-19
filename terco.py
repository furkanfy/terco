import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
df=pd.read_csv('C:/Users/ASUS/Documents/terco/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head())
print(df['Churn'])
df.info()

print(df.isnull().sum())

churn_counts = df['Churn'].value_counts()
print(churn_counts)

churn_percent = df['Churn'].value_counts(normalize=True) * 100
print("\nYuzde dagilimi:")
print(churn_percent)
df = df.drop('customerID', axis=1)  # customerID'yi siliyoruz

df = pd.get_dummies(df, columns=[
    'gender','Partner','Dependents', 'PhoneService',
    'MultipleLines','InternetService','OnlineSecurity',
    'OnlineBackup','DeviceProtection','TechSupport',
    'Churn'  # bunu da sayısala çevirmemiz doğru, model için lazım
], drop_first=True)

df.columns = df.columns.str.strip()
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı:", accuracy)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))