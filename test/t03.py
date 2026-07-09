# 导入 pandas 库，用于数据处理和统计分析
import pandas as pd
# 导入 numpy 库，用于生成随机数和数值计算
import numpy as np

# ==========================================
# 1. 创建随机数据
# ==========================================
# 固定随机种子为 42，保证每次运行生成的随机数完全相同，便于结果复现
RandomSeed = 42
np.random.seed(RandomSeed)
# 用字典定义两列数据：
#   Score: 从正态分布中采样，均值 70，标准差 10，生成 100 个样本
#          np.random.normal(loc, scale, size)
#            loc    = 70  → 分布的中心（均值）
#            scale  = 10  → 分布的宽度（标准差）
#            size   = 100 → 生成样本数量
#   Hours_Studied: 从均匀分布中采样，范围 [1, 10)，生成 100 个样本
#          np.random.uniform(low, high, size)
#            low  = 1   → 最小值（包含）
#            high = 10  → 最大值（不包含）
#            size = 100 → 生成样本数量
data = {
    'Score': np.random.normal(70, 10, 100),
    'Hours_Studied': np.random.uniform(1, 10, 100)
}

# 将字典转换为 DataFrame 格式（类似 Excel 表格），自动以键作为列名
df = pd.DataFrame(data)

# ==========================================
# 2. 查看基本统计信息
# ==========================================
# df.describe() 会输出每列的 8 项统计摘要：
#   count  - 样本数量
#   mean   - 均值
#   std    - 标准差
#   min    - 最小值
#   25%    - 第一四分位数（25% 的数据低于此值）
#   50%    - 中位数（50% 的数据低于此值）
#   75%    - 第三四分位数（75% 的数据低于此值）
#   max    - 最大值
print(df.describe())


# ==========================================
# 3. 计算 Score 列的中位数
# ==========================================
# df['Score'].median() 返回 Score 列的中位数
# 中位数：将数据从小到大排序后，位于正中间的值
# 相比均值，中位数对极端值（异常值）更鲁棒，不易被拉偏
print("Median Score:", df['Score'].median())

# ==========================================
# 4. 计算两列之间的相关系数
# ==========================================
# df['Score'].corr(df['Hours_Studied']) 计算 Pearson 相关系数
# 取值范围：[-1, 1]
#   接近  1  → 强正相关（一个增大，另一个也增大）
#   接近 -1  → 强负相关（一个增大，另一个减小）
#   接近  0  → 无线性相关
#
# 注意：这里的 Score 和 Hours_Studied 是独立生成的随机数
# 所以相关系数应该接近 0，表示两者没有线性关系
print("Correlation:", df['Score'].corr(df['Hours_Studied']))



# 指定低值和高值，同时指定大小
arr2 = np.random.uniform(low=-5.0, high=5.0, size=(2, 3))
print("Random Array 2:\n", arr2)
