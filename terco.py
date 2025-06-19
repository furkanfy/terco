import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
df=pd.read_csv('C:/Users/ASUS/Documents/terco/WA_Fn-UseC_-Telco-Customer-Churn.csv')

#print(df.head())
#print(df['Churn'])
df.info()

#print(df.isnull().sum())

churn_counts = df['Churn'].value_counts()
#print(churn_counts)

churn_percent = df['Churn'].value_counts(normalize=True) * 100
#print("\nYuzde dagilimi:")
#print(churn_percent)
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

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=400,max_depth=50)
model=forest.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
#print("Doğruluk Oran:", accuracy)
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5,
                           n_informative=3, n_redundant=0,
                           weights=[0.9], random_state=42)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

#print("Eski sinif dağilimi:", dict(pd.Series(y).value_counts()))
#print("Yeni sinif dağilimi:", dict(pd.Series(y_res).value_counts()))


model = RandomForestClassifier(n_estimators=300, max_depth=50, random_state=42)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)

#print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# Tahmin olasılıkları (Churn = Yes için)
y_proba = model.predict_proba(X_test)[:,1]

# ROC eğrisi
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Çizim
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi - Random Forest (SMOTE ile)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

feature_importances = pd.DataFrame({'Feature': X_train.columns,
                                    'Importance': model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# En önemli 10 özelliği çizdir
plt.figure(figsize=(10,6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10], color='teal')
plt.gca().invert_yaxis()
plt.title('En Önemli 10 Özellik')
plt.xlabel('Önem Skoru')
plt.grid(True)
plt.show()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='recall',  # churn=Yes sınıfı önemli olduğu için recall'ı artırmak istiyoruz
    n_jobs=-1,  # tüm çekirdekleri kullan
    verbose=2
)

grid_search.fit(X_res, y_res)
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi recall skoru:", grid_search.best_score_) 


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# ROC AUC
y_proba_best = best_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_proba_best)
print("Yeni ROC AUC Skoru:", roc_auc)

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

importance = xgb.feature_importances_
features = X_train.columns

# Görselleştirme
plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=features)
plt.title("Özellik Önem Düzeyi (Feature Importance)")
plt.xlabel("Önem")
plt.ylabel("Özellikler")
plt.tight_layout()
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(eval_metric='logloss'))
])
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=3)
grid.fit(X_train, y_train)
print("En iyi parametreler:", grid.best_params_)
print("En iyi skor:", grid.best_score_)

import joblib

# Eğitilmiş GridSearchCV pipeline'ı kaydet
joblib.dump(grid.best_estimator_, 'xgb_pipeline.pkl')
# Modeli yükle
model = joblib.load('xgb_pipeline.pkl')

# Yeni veri tahmini (örnek veriyle)
sample = X_test.iloc[0:1]
prediction = model.predict(sample)
print("Tahmin:", prediction)