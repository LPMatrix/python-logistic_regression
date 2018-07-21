import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ads = pd.read_csv('advertising.csv')

sns.countplot(data=ads,x='Male',hue='Clicked on Ad')
plt.show()

X = ads.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis=1)
y = ads['Clicked on Ad']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
lr = LogisticRegression()
lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

print(classification_report(y_test,predictions))