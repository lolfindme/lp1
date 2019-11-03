import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url="/home/gmail/Downloads/BigMartTrain.csv"
url1="/home/gmail/Downloads/BigMartTest.csv"
train=pd.read_csv(url)
test=pd.read_csv(url1)
#print(df.describe())

train['source']='Train'
test['source']='Test'

df=pd.concat( [train,test],ignore_index=True)

avg1=df["Item_Weight"].mean(axis=0)
df["Item_Weight"].replace(np.nan,avg1,inplace=True)
#print(df.head(10))

avg2=df["Item_Visibility"].mean(axis=0)
df["Item_Visibility"].replace(0,avg2,inplace=True)
#print(df.head(10))




df["Item_Fat_Content"].replace(['LF', 'low fat', 'Low fat','Low Fat'],0,inplace=True)
df["Item_Fat_Content"].replace(['Regular','reg'],1,inplace=True)
#print(df["Item_Fat_Content"])




outlet_size_mode=df.pivot_table(values='Outlet_Size',columns='Outlet_Location_Type',aggfunc=(lambda x:x.mode().iat[0]))
#print(outlet_size_mode)

miss_bool=df["Outlet_Size"].isnull()
df.loc[miss_bool,'Outlet_Size']=df.loc[miss_bool,'Outlet_Location_Type'].apply(lambda x:outlet_size_mode[x])
#print(df["Outlet_Size"])




varlist=['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Type','Item_Fat_Content']
dataf=pd.get_dummies(df,columns=varlist)
dataf.head(10)





train=dataf.loc[dataf['source']=='Train']
test=dataf.loc[dataf['source']=='Test']
x_train=train.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','source','Item_Outlet_Sales'],axis=1)
y_train=train['Item_Outlet_Sales']
x_test=test.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','source','Item_Outlet_Sales'],axis=1)
y_test=test['Item_Outlet_Sales']
y_train.head(10)




algo= LinearRegression(normalize="True")
algo.fit(x_train,y_train)
print(algo.intercept_)
print(algo.coef_)

train_predict=algo.predict(x_train)
print(train_predict)

rmse=mean_squared_error(y_train,train_predict)
print(rmse)


test_predict= algo.predict(x_test)
print(test_predict)
"""administrator@112A-08:~$ cd demo
administrator@112A-08:~/demo$ python3 as3.py
1.6349205988127442e+16
[-4.04709245e-01 -1.72832353e+02  1.56239324e+01 -7.80592153e+15
 -7.80592153e+15 -7.80592153e+15  1.52871123e+16  1.52871123e+16
  1.52871123e+16 -1.79755361e+16 -1.79755361e+16 -1.79755361e+16
 -1.79755361e+16 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15
 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15
 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15
 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15 -4.78712806e+15
 -4.78712806e+15 -1.06773259e+15 -1.06773259e+15]
[4032.  576. 2396. ... 1432. 1410. 1216.]
1273525.553874833
[1840. 1478. 1914. ... 1948. 3524. 1412.]
administrator@112A-08:~/demo$ 

"""
