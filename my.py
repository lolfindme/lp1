
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
data = pd.read_csv("iris.csv") 
print (data.head(10)) 
print (data.describe() ) 
print(data.info())
plt.figure(figsize = (10, 7)) 
x = data.sepal_width
plt.hist(x, bins = 20, color = "green") 
plt.title("Sepal Width in cm") 
plt.xlabel("Sepal_Width_cm") 
plt.ylabel("Count") 
plt.show() 
plt.figure(figsize = (10, 7)) 
x = data.sepal_length
plt.hist(x, bins = 20, color = "blue") 
plt.title("Sepal Length in cm") 
plt.xlabel("Sepal_Length_cm") 
plt.ylabel("Count") 
plt.show() 
new_data = data[["sepal_length", "sepal_width","petal_width","petal_length"]] 
print(new_data.head()) 
plt.figure(figsize = (20, 7)) 
new_data.boxplot() 
plt.show()

