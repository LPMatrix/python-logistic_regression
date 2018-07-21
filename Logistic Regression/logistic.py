import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train = pd.read_csv('titanic_train.csv')

sns.heatmap(train.isnull(),cbar=False,yticklabels=False)
plt.show()

sns.countplot(data=train, x='Survived', hue='Sex')
plt.show()

sns.countplot(data=train,x='Survived',hue='Pclass')
plt.show()

train.drop('Cabin',inplace=True,axis=1)

train['Age'].fillna(value=train['Age'].mean(), inplace=True)

train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)

train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

X = train.drop('Survived',axis=1)
y = train['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)
LRM = LogisticRegression()
LRM.fit(X_train,y_train)
predictions = LRM.predict(X_test)

print(classification_report(y_test,predictions))

plt.scatter(y_test,predictions)
plt.show()