# 导入 pandas 库，用于数据处理和创建 DataFrame
import pandas as pd
# 导入 StandardScaler 类，用于数据的标准化（Z-Score 标准化）
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 创建原始数据
# ==========================================
# 用字典定义数据：键为列名，值为该列的数据列表
# 注意：每个列表的长度必须一致，否则 pd.DataFrame 会报错
data = {
    'Height': [165, 175, 180, 160, 170],  # 身高（cm），共 5 个样本
    'Weight': [60, 70, 80, 55, 65]        # 体重（kg），共 5 个样本
}

# 将字典转换为 DataFrame 格式（类似 Excel 表格）
# pd.DataFrame(data) 会自动把字典的键作为列名，值作为列数据
df = pd.DataFrame(data)

# ==========================================
# 2. 初始化标准化器
# ==========================================
# StandardScaler() 执行 Z-Score 标准化
# 公式：z = (x - mean) / std
# 其中 mean 是列均值，std 是列标准差
# 标准化后：数据均值变为 0，标准差变为 1
scaler = StandardScaler()

print("原始数据：\n",df) 
print("数据标准化器：\n",scaler)
print("数据标准化器参数：\n",scaler.get_params())

# ==========================================
# 3. 对数据进行标准化
# ==========================================
# scaler.fit_transform(df) 分两步执行：
#   fit()：计算每列的均值和标准差（记住这些统计量）
#   transform()：用计算出的均值和标准差对数据进行标准化转换
#   fit_transform() 是两者的合并调用，更简洁
#
# 返回值是一个 Numpy 二维数组，需要用 pd.DataFrame() 转回表格格式
#
# columns=df.columns 参数说明：
#   保留原始 DataFrame 的列名（'Height', 'Weight'）
#   如果不指定，列名会丢失，变成默认的 0, 1, 2...
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


print("数据scaler：\n",scaler.fit_transform(df))
# 输出标准化后的结果
# 每个数值表示原始值偏离均值几个标准差
# 例如 1.2 表示该值比均值高 1.2 个标准差
print("标准化后数据：\n",df_scaled)


