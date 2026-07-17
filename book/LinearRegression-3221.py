# ==========================================
# 1. 导入必要的库
# ==========================================
# 逻辑回归：用于解决二分类问题，本例根据年龄和薪水预测用户是否购买商品
import numpy as np                 # 数值计算库：用于生成网格点、数组变换等操作
import matplotlib.pyplot as plt    # 数据可视化库：用于绘制分类边界和散点图
import pandas as pd                # 数据处理库：用于读取和操作表格数据（CSV）
from sklearn.model_selection import train_test_split # 导入数据集拆分工具
from sklearn.preprocessing import StandardScaler     # 导入特征标准化工具
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归分类模型
from sklearn.metrics import confusion_matrix         # 导入混淆矩阵评估工具
from matplotlib.colors import ListedColormap         # 导入颜色映射工具，用于自定义图表颜色

# ==========================================
# 2. 导入数据集 (Data Preprocessing)
# ==========================================
# 读取本地名为 Social_Network_Ads.csv 的数据文件
dataset = pd.read_csv('./book/Social_Network_Ads.csv')

# 提取特征（自变量 X）：取所有行 (:)，取第 2 列和第 3 列（索引为 2、3，即年龄和薪水）
# .values 的作用是将 Pandas 的 DataFrame 格式转换成 Numpy 数组，方便 Sklearn 处理
X = dataset.iloc[:, [2, 3]].values

# 提取目标变量（因变量 y）：取所有行 (:)，取第 4 列（索引为 4，即是否购买）
# 目标值通常为 0 或 1，表示两个不同的分类结果
y = dataset.iloc[:, 4].values

# ==========================================
# 3. 划分训练集和测试集 (Split dataset)
# ==========================================
# train_test_split 用于将数据随机打乱并按照比例拆分
# test_size=0.25 ：表示测试集占总数据量的 25%（剩下的 75% 作为训练集）
# random_state=0 ：设置随机种子，保证每次运行时拆分结果一致，便于结果复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ==========================================
# 4. 特征尺度调整 (Feature Scaling)
# ==========================================
# StandardScaler 会将特征转换为均值为 0、标准差为 1 的标准化数据
# 因为年龄和薪水的数值范围差距较大，标准化可以避免薪水对模型训练产生过大影响
sc = StandardScaler()

# fit_transform：先根据训练集计算均值和标准差，再对训练集进行标准化
X_train = sc.fit_transform(X_train)

# transform：使用训练集得到的均值和标准差，对测试集做同样的标准化
# 注意：测试集不能重新 fit，否则会引入测试数据的信息
X_test = sc.transform(X_test)

# ==========================================
# 5. 训练模型 (Train the model)
# ==========================================
# 实例化逻辑回归分类器
classifier = LogisticRegression(random_state=0)

# 使用训练集数据（X_train, y_train）来训练（拟合）这个分类模型
# 执行这一步后，模型会学习出一条分类边界，用来区分是否购买
classifier.fit(X_train, y_train)

# ==========================================
# 6. 预测测试集结果 (Make predictions)
# ==========================================
# 使用刚才训练好的分类器，对没见过的测试集特征（X_test）进行预测
# 得到的结果 y_pred 是一组预测分类值（0 或 1）
y_pred = classifier.predict(X_test)

# ==========================================
# 7. 创建混淆矩阵 (Confusion Matrix)
# ==========================================
# confusion_matrix 用于比较真实结果 y_test 和预测结果 y_pred
# 可以直观看出模型预测正确和预测错误的样本数量
cm = confusion_matrix(y_test, y_pred)

# ==========================================
# 8. 训练集结果的可视化 (Visualize Training Set)
# ==========================================
# 将训练集特征和标签赋值给 X_set、y_set，便于后续统一绘图
X_set, y_set = X_train, y_train

# np.meshgrid 用于生成二维坐标网格
# 这里根据年龄和薪水两个标准化后的特征范围，生成密集网格点，用来绘制分类区域背景
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# 对网格中每个点进行预测，并使用不同颜色填充分类区域
# ravel() 将二维网格展开为一维数组；reshape(X1.shape) 再把预测结果还原为网格形状
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('black', 'gray')))

# 设置 X 轴和 Y 轴的显示范围，与生成的网格范围保持一致
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 绘制训练集中的真实样本点
# 不同类别（0 或 1）使用不同颜色显示，并通过 label 显示图例
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('black', 'gray'))(i), label=j)

# 以下两行是为了解决 Matplotlib 在画图时无法显示中文（出现方块乱码）的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加图表的标题、X轴和Y轴标签
plt.title('逻辑回归（训练集）')
plt.xlabel('年龄')
plt.ylabel('薪水')

# 显示图例和图表
plt.legend()
plt.show()

# ==========================================
# 9. 测试集结果的可视化 (Visualize Test Set)
# ==========================================
# 将测试集特征和标签赋值给 X_set、y_set
X_set, y_set = X_test, y_test

# 生成测试集范围内的二维坐标网格，用于绘制测试集上的分类区域背景
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# 对测试集网格点进行预测，并填充不同类别对应的背景区域
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('black', 'gray')))

# 设置坐标轴显示范围
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 绘制测试集中的真实样本点，用于观察模型在未见过数据上的分类效果
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('black', 'gray'))(i), label=j)

# 再次设置中文字体（为了保险，很多教程会在画新图时重复设置一遍）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表信息
plt.title('逻辑回归（测试集）')
plt.xlabel('年龄')
plt.ylabel('薪水')

# 显示图例和图表
plt.legend()
plt.show()
