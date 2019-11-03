import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv('/home/gmail/Downloads/pima_indians_diabetes.csv')
print(data.head(10))

x=data.drop(['1'],axis=1)
y=data['1']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)

obj = GaussianNB()
obj.fit(x_train, y_train)
pred = obj.predict(x_test)
print(accuracy_score(y_test,pred))




