import pandas as pd
import numpy as np
#创建包含缺失值的DataFrame
data={'A': [1,2,np.nan,4],
'B': [5, np.nan, np.nan,8],
'C': [9, 10, 11, 12]}

print("---------------------------------------------------data")
print(data)

df = pd.DataFrame(data)
print("---------------------------------------------------df")
print(df)
# 查看缺失值数量、
print("---------------------------------------------------df.isnull()")
print(df.isnull())
print("---------------------------------------------------print(df.isnull().sum())")
print("缺失值数量:",df.isnull().sum())
print("-------------------------------------------------print(df.isnull().sum().sum())")
print("缺失值数量:",df.isnull().sum().sum())
# 删除所有含有缺失值的行
df_drop = df.dropna()
#用列的平均值填充缺失值
df_fill = df.fillna(df.mean())
#打印处理结果(辅助验证)
print("---------------------------------------------------")
print("删除缺失值后的行数:",df_drop.shape[0])
print("填充后是否还有缺失值:",df_fill.isnull().any().any())
print("---------------------------------------------------df_fill")
print(df_fill)
print("---------------------------------------------------")