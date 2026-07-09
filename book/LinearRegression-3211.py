# ==========================================
# 1. 导入必要的库
# ==========================================
import pandas as pd               # 数据处理库：用于读取和操作表格数据（CSV）
import matplotlib.pyplot as plt   # 数据可视化库：用于画图
from sklearn.linear_model import LinearRegression    # 导入线性回归模型
from sklearn.model_selection import train_test_split # 导入数据集拆分工具
import os

# ==========================================
# 2. 导入数据集 (Data Preprocessing)
# ==========================================
# 读取本地名为 Salary_Data.csv 的数据文件（假设文件在代码同目录下）
# dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Salary_Data.csv'))
dataset = pd.read_csv('./book/Salary_Data.csv')

# 提取特征（自变量 X）：取所有行 (:)，以及除了最后一列之外的所有列 (:-1)
# .values 的作用是将 Pandas 的 DataFrame 格式转换成 Numpy 数组，方便 Sklearn 处理
X = dataset.iloc[:, :-1].values  

# 提取目标变量（因变量 y）：取所有行 (:)，取第 1 列（即第2列，索引为1，也就是薪资）
# 注意：Python 的索引是从 0 开始的
y = dataset.iloc[:, 1].values    

# ==========================================
# 3. 划分训练集和测试集 (Split dataset)
# ==========================================
# train_test_split 用于将数据随机打乱并按照比例拆分
# test_size=1/3 ：表示测试集占总数据量的三分之一（剩下的三分之二作为训练集）
# random_state=0 ：设置随机种子。保证每次运行代码时，拆分的训练/测试数据都是完全一样的，便于结果复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# ==========================================
# 4. 训练模型 (Train the model)
# ==========================================
# 实例化线性回归对象
regressor = LinearRegression()

# 使用训练集数据（X_train, y_train）来训练（拟合）这个线性模型
# 执行这一步后，模型内部会计算出一条最佳的直线方程：y = ax + b
regressor.fit(X_train, y_train)

# ==========================================
# 5. 预测测试集结果 (Make predictions)
# ==========================================
# 使用刚才训练好的模型，对没见过的测试集特征（X_test）进行预测
# 得到的结果 y_pred 是一组预测的薪资数值
y_pred = regressor.predict(X_test)

# ==========================================
# 6. 训练集结果的可视化 (Visualize Training Set)
# ==========================================
# 以下两行是为了解决 Matplotlib 在画图时无法显示中文（出现方块乱码）的问题
plt.rcParams["font.sans-serif"] = ['SimHei']   # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False     # 解决负号 '-' 显示为方块的问题

# 绘制训练集的真实数据点（红色的散点）
# x轴是工作年限(X_train)，y轴是真实薪资(y_train)
plt.scatter(X_train, y_train, color='red')

# 绘制训练集上模型拟合的直线（蓝色的实线）
# 这里重新调用 regressor.predict(X_train) 获取训练集预测值，并连线
plt.plot(X_train, regressor.predict(X_train), color='blue')

# 添加图表的标题、X轴和Y轴标签
plt.title('Salary vs Years of Experience (Training Set)') # 标题
plt.xlabel('Years of Experience') # X轴标签：工作年限
plt.ylabel('Salary')              # Y轴标签：薪水

# 显示图表
plt.show()

# ==========================================
# 7. 测试集结果的可视化 (Visualize Test Set)
# ==========================================
# 再次设置中文字体（为了保险，很多教程会在画新图时重复设置一遍）
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# 绘制测试集的真实数据点（红色的散点）
# 注意：这里用的是 X_test 和 y_test
plt.scatter(X_test, y_test, color='red')

# 绘制拟合直线（蓝色的实线）
plt.plot(X_train, regressor.predict(X_train), color='blue')



# 设置图表信息
plt.title('Salary vs Years of Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# 显示图表
plt.show()