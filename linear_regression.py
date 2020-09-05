from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import Dataset from sklearn 
from sklearn.datasets import load_iris
%matplotlib inline
iris=load_iris()
y=iris.target
y

iris.target_names

x=iris.data
x

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)
clf=LinearRegression()
clf.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_pred=clf.predict(x_test)
y_pred

#sepal length
mean_x=x_test[:,0].mean()
mean_y=y_pred.mean()
#calculating B1 and B0
b1=np.divide(np.sum(np.multiply(np.subtract(x_test[:,0],mean_x),np.subtract(y_pred,mean_y))),np.sum(np.square(np.subtract(x_test[:,0],mean_x))))
print("B1: ",b1)
b0=np.subtract(mean_y,np.multiply(b1,mean_x))
print("B0: ",b0)
plt.scatter(x_test[:,0],y_pred)
plt.show()

#sepal width
mean_x=x_test[:,1].mean()
mean_y=y_pred.mean()
#calculating B1 and B0
b1=np.divide(np.sum(np.multiply(np.subtract(x_test[:,1],mean_x),np.subtract(y_pred,mean_y))),np.sum(np.square(np.subtract(x_test[:,1],mean_x))))
print("B1: ",b1)
b0=np.subtract(mean_y,np.multiply(b1,mean_x))
print("B0: ",b0)
plt.scatter(x_test[:,1],y_pred)
plt.show()

#petal length
mean_x=x_test[:,2].mean()
mean_y=y_pred.mean()
#calculating B1 and B0
b1=np.divide(np.sum(np.multiply(np.subtract(x_test[:,2],mean_x),np.subtract(y_pred,mean_y))),np.sum(np.square(np.subtract(x_test[:,2],mean_x))))
print("B1: ",b1)
b0=np.subtract(mean_y,np.multiply(b1,mean_x))
print("B0: ",b0)
plt.scatter(x_test[:,2],y_pred)
plt.show()


#petal width
mean_x=x_test[:,3].mean()
mean_y=y_pred.mean()
#calculating B1 and B0
b1=np.divide(np.sum(np.multiply(np.subtract(x_test[:,3],mean_x),np.subtract(y_pred,mean_y))),np.sum(np.square(np.subtract(x_test[:,3],mean_x))))
print("B1: ",b1)
b0=np.subtract(mean_y,np.multiply(b1,mean_x))
print("B0: ",b0)
plt.scatter(x_test[:,3],y_pred)
plt.show()

#Y=b0+b1*X
b0:  -0.05834498788583135
b1:  3.418471054031162
prediction=b0+b1*x_test
prediction
#prediction will be compared to y_test for error

rmse1=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction[:,0],y_test))),x.size))
#print("Root mean squared error is for sepal length vs sepal width is: ",rmse1)
rmse2=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction[:,1],y_test))),x.size))
#print("Root mean squared error is for sepal length vs sepal width is: ",rmse2)
rmse3=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction[:,2],y_test))),x.size))
#print("Root mean squared error is for sepal length vs sepal width is: ",rmse3)
rmse4=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction[:,3],y_test))),x.size))
#print("Root mean squared error is for sepal length vs sepal width is: ",rmse4)
rmse=(rmse1+rmse2+rmse3+rmse4)/4
print("Root mean squared error is for sepal length vs sepal width",rmse)


//NON LINEAR CURVES
prediction=b0+b1*np.square(x_test)
rmse=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction[:,3],y_test))),x.size))
print("Root mean squared error is for sepal length vs sepal width is using non linear curve: ",rmse)
