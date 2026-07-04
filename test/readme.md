这份文档是关于 **GMDH（数据分组处理方法，一种多项式神经网络算法）** 进行数据预测、特征选择和相关可视化的完整案例。

---

# 【案例】GMDH数据预测

> **⚠️ 环境说明：** 截图中使用了 `from gmdh import Gmdh`，这通常需要安装第三方库 `pygmdh` 或类似的 GMDH 算法包。如您尚未安装，可使用 `pip install pygmdh` 尝试安装，或根据具体环境安装对应的GMDH库。

## 一、GMDH数据预测

### 1. 导入相关的库
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

### 2. 加载数据集及基本数据处理
```python
# 1、加载数据集
dataset = pd.read_csv('GMDH.csv')

# 2、划分特征和目标变量
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# 3、查看数据集基本信息
print("数据形状:", dataset.shape)
print(dataset.head())

# 4、查看缺失值情况
print("缺失值情况:", dataset.isnull().sum())
```

### 3. 数据预处理（缺失值填补）
*说明：由于数据中可能存在缺失值，使用均值填补策略。*
```python
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = y.values.reshape(-1, 1)
y = imputer.fit_transform(y)
y = y.ravel()
```

### 4. 划分训练集和测试集
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

### 5. 特征缩放（标准化）
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. 模型构建与预测
```python
from gmdh import Gmdh
model = Gmdh()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 7. 模型评估
```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('GMDH 模型评估结果:')
print('R2:', r2)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
```

---

## 二、相关性分析

### 1. 代码实现
```python
# 导入相关库（同上）
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据及处理（同上）
dataset = pd.read_csv('GMDH.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = y.values.reshape(-1, 1)
y = imputer.fit_transform(y)
y = y.ravel()

# 2. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 3. 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 计算特征相关性矩阵并可视化
# 将numpy数组重新转化为DataFrame以便获取列名
corr_matrix = pd.DataFrame(X_train, columns=dataset.columns[:-1]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('特征相关性热力图')
plt.show()
```

### 2. 分析说明
* 通过热力图可以直观地查看各特征之间的相关系数大小。
* **数值越大，说明两个特征之间的相关性越强。**
* 若存在高度共线性的特征（相关系数接近 1 或 -1），通常需要根据业务需求或使用 Lasso、GMDH 等算法进行特征剔除或降维。

---

## 三、Lasso回归模型

*使用 Lasso 回归替代 GMDH，进行对比或降维选择。*
```python
# 导入库与数据预处理流程同上
# ...
# 模型构建与训练
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 预测与评估
y_pred_lasso = lasso.predict(X_test)
print('Lasso 回归模型评估结果:')
print('R2:', r2_score(y_test, y_pred_lasso))
print('MAE:', mean_absolute_error(y_test, y_pred_lasso))
print('MSE:', mean_squared_error(y_test, y_pred_lasso))
```

---

## 四、GMDH与变量选择

*GMDH 算法不仅具备预测能力，还具有自动筛选重要特征的特性。*
```python
# 导入相关的库
from gmdh import Gmdh

# 同样的数据预处理、划分、标准化流程（略）
# ...

# 构建并训练 GMDH 模型
model = Gmdh()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 输出 GMDH 模型所选择的特征变量索引
print("GMDH 模型选择的变量索引：", model.get_selected_indices())
print('GMDH 模型评估结果:')
print('R2:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
```

---

## 五、特征相关性可视化

*在建模前，对特征进行探索性数据分析（EDA）以了解数据分布。*
```python
# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据框用于画图
df = pd.DataFrame(X_train, columns=dataset.columns[:-1])

# 1. 绘制箱线图（查看数据分布和离群点）
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('特征数据分布箱线图')
plt.show()

# 2. 散点图可视化（示例：任意取两个特征进行可视化）
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.5, color='blue')
plt.title('前两个特征的散点图')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.show()
```

---

## 六、GMDH模型与变量选择

*此部分为重复练习或针对特定数据集（本案例截图中为 GMDH.csv）的运行示范。*
```python
# 导入库与预处理
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from gmdh import Gmdh
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. 数据加载与划分
dataset = pd.read_csv('GMDH.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = y.reshape(-1, 1)
y = imputer.fit_transform(y)
y = y.ravel()

# 2. 训练集测试集划分及标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. GMDH 模型构建
model = Gmdh()
model.fit(X_train, y_train)

# 4. 预测与评估
y_pred = model.predict(X_test)
print('GMDH 评估结果:')
print(f'R2: {r2_score(y_test, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')
print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')
```

> **📌 附加提示：** 大部分机器学习代码开始前，均执行了 `SimpleImputer` 填补缺失值、`train_test_split` 划分数据、`StandardScaler` 标准化数据三大步骤。这是非常标准、规范的机器学习数据预处理流水线（Pipeline）。