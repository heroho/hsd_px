# Pandas 的 DataFrame 是数据分析中最核心的数据结构。下面我将按使用场景为你梳理最常用的方法，并附上关键代码示例和避坑指南。

# 1. 数据概览与检查（EDA 必备）
# 这几个方法能让你快速“摸清”数据底细。

# df.head(n) / df.tail(n)：查看前/后 n 行（默认 5 行），快速浏览数据样貌。

# df.info()：查看列名、非空值数量、每列数据类型（Dtype），诊断缺失值首选。

# df.describe(include='all')：对数值列生成统计摘要（均值、分位数等）；对文本列显示频次最高的值。

# df.shape / df.columns / df.index：属性（非方法），分别返回行列数、列名列表和行索引。

# df.sample(n)：随机抽取 n 行，适合抽样验证。

# 2. 数据选取与过滤（增删改查）
# 核心在于区分 标签索引 和 位置索引。

# 方法/属性	说明	示例
# df[col] / df[[col1, col2]]	单列选取 / 多列子集	df['年龄']
# df.loc[行标签, 列标签]	基于标签，包含结束边界	df.loc[0:5, '姓名':'年龄']
# df.iloc[行位置, 列位置]	基于位置（0开始），不包含结束边界	df.iloc[:, 0:3]
# df.at[行标签, 列名]	快速访问单个值（比 loc 更快）	df.at[0, '姓名']
# df.iat[行位置, 列位置]	快速访问单个值（比 iloc 更快）	df.iat[0, 1]
# df.query('条件表达式')	使用字符串表达式过滤，书写简洁	df.query('年龄 > 18 and 城市=="北京"')
# df.isin([值列表])	判断是否包含列表中的值，常用于筛选	df[df['ID'].isin([1001, 1002])]
# 3. 数据清洗（高频操作）
# 实战中 80% 的时间都在处理脏数据。

# 处理缺失值：df.isnull() / df.notnull()（判断）；df.dropna(axis=0, subset=['列名'])（删除缺失行）；df.fillna(value={'列A': 0, '列B': '未知'})（填充缺失）。

# 处理重复值：df.duplicated(subset=['列1'])（标记重复）；df.drop_duplicates(subset=['列1'], keep='first')（删除重复，保留首次出现）。

# 列名与类型：df.rename(columns={'旧名':'新名'}, inplace=True)（重命名）；df.astype({'列名': 'str'})（强制类型转换）。

# 替换值：df.replace({'旧值': '新值'}) 或针对正则：df.replace(regex={r'\D': ''}, value='')。

# 4. 排序与排名
# df.sort_values(by=['列A', '列B'], ascending=[True, False])：按指定列排序，by 支持多列。

# df.sort_index(axis=0, ascending=False)：按行索引或列索引排序。

# df.rank(method='dense')：计算排名，适合竞赛评分场景。

# 5. 分组聚合（分析核心）
# df.groupby('分组列')['数值列'].sum() / mean() / count()：基础分组统计。

# df.groupby(['A','B']).agg( {'C': 'sum', 'D': lambda x: x.max() - x.min()} )：agg() 非常灵活，可对不同列指定不同聚合函数。

# df.pivot_table(index='行索引', columns='列分类', values='值', aggfunc='mean', fill_value=0)：透视表，类似 Excel 数据透视表，支持多级分组和填充。

# df.groupby('列').transform(lambda x: x - x.mean())：保留原行数返回聚合结果（去中心化），常用于特征工程。

# 6. 函数式操作（批量处理）
# df.apply(func, axis=0)：将函数应用到行（axis=1）或列（axis=0）。例如：df.apply(lambda row: row['A'] + row['B'], axis=1)。

# df.applymap(func)：将函数应用到每一个元素，常用于数据格式化（如保留两位小数）。

# df.pipe(func)：链式调用，将整个 df 传入函数，避免中间变量污染。

# 7. 表间操作（拼接与连接）
# pd.concat([df1, df2], axis=0, ignore_index=True)：纵向堆叠（增加行），建议设置 ignore_index 重置索引。

# df1.merge(df2, on='关联键', how='inner/left/right/outer')：SQL 风格连接，按列匹配。how 参数决定保留哪些数据。

# df1.join(df2, on='索引或列', how='left')：基于索引的连接，比 merge 更轻量。

# 8. 输入输出（IO）
# df.to_csv('path.csv', index=False, encoding='utf-8-sig')：导出 CSV，index=False 防止多出一列索引。

# df.to_excel('path.xlsx', sheet_name='Sheet1', index=False)：需安装 openpyxl。

# 读取时对应：pd.read_csv() / pd.read_excel()。

# ⚠️ 资深避坑指南（重要）
# 视图 vs 副本：df[df['A']>1]['B'] = 0 会触发 SettingWithCopyWarning。正确做法：要么用 df.loc[df['A']>1, 'B'] = 0，要么显式使用 df.copy()。

# inplace=True 的取舍：官方建议逐步弃用。推荐返回新对象（df_new = df.dropna()），便于链式操作和调试，避免副作用。

# 内存优化：当数据很大时，读取时指定 dtypes（如 float32 替代 float64），或使用 df.memory_usage(deep=True) 排查内存占用。



import pandas as pd
import numpy as np

data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12],
        'D': [13, 14, 15, np.nan],
        'E': [np.nan, 17, 18, 19],
        'F': [20, 21, 22, 23],
        'G': [24, 25, 26, 27]
        }
df = pd.DataFrame(data)
# print("DataFrame:\n", df)
# print("Null values:\n", df.isnull())
# print("Null values count:\n", df.isnull().sum())
# print("Total null values count:", df.isnull().sum().sum)
# print("Fill null values with 0:\n", df.fillna(0))
# print("Fill null values with mean:\n", df.fillna(df.mean()))
# print("First 5 rows:\n", df.head(2))
# print("Last 5 rows:\n", df.tail(2))
# print("DataFrame Info:\n", df.info())
# print("Describe:\n", df.describe())
# print("Describe all:\n", df.describe( include="all"))

# print("DataFrame Shape:", df.shape)
# print("DataFrame Columns:", df.columns)
# print("DataFrame Index:", df.index)
# print("Sampled Rows:\n", df.sample(n=2, random_state=42))  # Randomly sample 2 rows from the DataFrame
# df['D'] = df['A'] + df['B']
# print("DataFrame:\n", df)

# print("DataFrame   df['A','C']:\n",  df[['A','C']])
# df.loc[0] = [1, 2, 3, 4, 5, 6, 7]
# print("DataFrame:\n", df)
# print("Selected Rows and Columns:\n", df.loc[:2, ['A', 'B']]) 
# print("Selected Rows and Columns (iloc):\n", df.iloc[1:2, 0:-2])
# print("Selected Value:\n", df.at[1, 'A'])
# print("Filtered DataFrame:\n", df.query('A > 1'))
# print("Filtered DataFrame (isin):\n", df[df['A'].isin([1, 2, 3])])
# print("Filtered DataFrame (isin):\n", df['A'].isin([1, 2, 3]))