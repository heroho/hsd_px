import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#导人数据集
dataset=pd.read_csv('./book/50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
print(dataset.head())

#对类别变量进行转换，编码为只含有0或1的虚拟变量
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
ct=ColumnTransformer([("state",OneHotEncoder(),[3])],remainder='passthrough')
X=ct.fit_transform(X)
#丢弃第一列，避免虚拟变量陷阱
X= X[:,1:]


#将数据集按比例拆分为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)
#调用简单线性回归算法拟合训练集
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#在测试集上预测结果
y_pred=regressor.predict(X_test)
print('Actual Values:' , y_test)
print( 'Predicted Values:' , y_pred)
