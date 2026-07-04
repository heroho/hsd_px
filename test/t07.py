# ==========================================
# 1. 导入必要的库
# ==========================================
# numpy：数值计算库，用于生成随机模拟数据
import numpy as np
# pandas：数据处理库，用于构建和存储 DataFrame
import pandas as pd
# LinearRegression：sklearn 的普通最小二乘法线性回归模型
from sklearn.linear_model import LinearRegression
# train_test_split：按比例拆分训练集和测试集
from sklearn.model_selection import train_test_split
# r2_score：计算决定系数 R²，衡量模型解释因变量变异的能力
from sklearn.metrics import r2_score

# ==========================================
# 2. 模拟数据生成
# ==========================================
# 固定随机种子为 56，保证每次运行生成的数据完全相同，便于结果复现
np.random.seed(56)

# n_samples：总共生成的样本数量（150 条航班记录）
n_samples = 150

# FlightDistance：航线距离（公里），从均匀分布 [500, 3000) 中随机采样
#   np.random.uniform(low, high, size)
#     low  = 500   → 最短航线 500 公里
#     high = 3000  → 最长航线 3000 公里
#     size = 150   → 生成 150 个样本
FlightDistance = np.random.uniform(500, 3000, n_samples)

# DepartureDelay：出发延误时间（分钟），从均匀分布 [0, 60) 中随机采样
#   0 分钟表示准点，60 分钟表示最大延误
DepartureDelay = np.random.uniform(0, 60, n_samples)

# noise：随机噪声，模拟现实中无法被模型解释的随机波动
#   np.random.normal(loc, scale, size)
#     loc  = 0   → 噪声均值为 0（不产生系统性偏差）
#     scale = 10  → 噪声标准差为 10（波动幅度）
#     size = 150  → 生成 150 个噪声值
noise = np.random.normal(0, 10, n_samples)

# ActualTime：实际飞行时间（分钟），由线性公式生成
# 公式：ActualTime = 40 + 0.12 * FlightDistance - 0.3 * DepartureDelay + noise
#   40                    → 基准飞行时间（分钟）
#   0.12 * FlightDistance → 航线越长，飞行时间越长（每公里增加 0.12 分钟）
#   -0.3 * DepartureDelay → 出发延误越多，实际飞行时间反而越短（模拟空中补偿效应）
#   noise                 → 叠加随机噪声，模拟真实数据的不确定性
ActualTime = 40 + 0.12 * FlightDistance - 0.3 * DepartureDelay + noise

# 构建 DataFrame：将三个特征列组合成表格形式
data = pd.DataFrame({
    'FlightDistance': FlightDistance,   # 航线距离
    'DepartureDelay': DepartureDelay,   # 出发延误
    'ActualTime': ActualTime            # 实际飞行时间（要预测的目标）
})

# 提取特征矩阵 X（双特征）和标签向量 y
# X：取 FlightDistance 和 DepartureDelay 两列，.values 转为 numpy 二维数组
#     形状为 (150, 2)，即 150 个样本 × 2 个特征
x = data[['FlightDistance', 'DepartureDelay']].values

# y：取 ActualTime 一列，.values 转为 numpy 一维数组
#     形状为 (150,)，即 150 个标签值
y = data["ActualTime"].values

# ==========================================
# 3. 数据集拆分（7:3 比例）
# ==========================================
# train_test_split 将数据随机打乱后按比例拆分
#   x              → 特征矩阵（150 × 2）
#   y              → 标签向量（150）
#   test_size=0.3  → 30% 作为测试集（45 条），70% 作为训练集（105 条）
#   random_state=56 → 固定随机种子，保证拆分结果可复现
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=56)

# ==========================================
# 4. 模型训练
# ==========================================
# LinearRegression() 实例化多特征线性回归模型
# 与 t04.py 的单特征不同，这里有两个特征，模型学习的是：
#   ActualTime = b + a1 * FlightDistance + a2 * DepartureDelay
#   b  = 截距（intercept_）
#   a1 = FlightDistance 的回归系数（coef_[0]）
#   a2 = DepartureDelay 的回归系数（coef_[1]）
model = LinearRegression()

# 用训练集数据拟合模型，计算最优的 b、a1、a2
model.fit(X_train, y_train)

# ==========================================
# 5. 预测与评估
# ==========================================
# 用测试集特征 X_test 进行预测
y_pred = model.predict(X_test)

# r2_score 计算决定系数 R²（Coeficient of Determination）
#   取值范围：0~1（越接近 1 表示模型越好）
#   R² = 1 - (残差平方和 / 总平方和)
#   可以理解为：模型解释了因变量多少比例的变异
#   例如 R²=0.98 表示模型解释了 98% 的变异，拟合非常好
r2 = r2_score(y_test, y_pred)

# ==========================================
# 6. 输出结果
# ==========================================
# f-string 格式化输出：
#   {model.intercept_:.2f}  → 截距，保留 2 位小数
#   {model.coef_[0]:.2f}    → FlightDistance 的回归系数
#   {model.coef_[1]:.2f}    → DepartureDelay 的回归系数
#   {r2:.4f}                → R² 分数，保留 4 位小数
print(f"回归方程:ActualTime = {model.intercept_:.2f} + {model.coef_[0]:.2f}xFlightDistance + {model.coef_[1]:.2f}xDepartureDelay")
print(f"模型决定系数(R2):{r2:.4f}")
