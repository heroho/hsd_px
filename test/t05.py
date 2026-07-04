# ==========================================
# 1. 导入必要的库
# ==========================================
# matplotlib.pyplot：Python 最基础的绑图库，用于创建和定制图表
import matplotlib.pyplot as plt
# pandas：数据处理库，用于读取 CSV 和操作 DataFrame
import pandas as pd
# seaborn：基于 matplotlib 的高级可视化库，提供更美观的统计图表
import seaborn as sns

# ==========================================
# 2. 加载数据
# ==========================================
# pd.read_csv() 读取 CSV 文件并返回 DataFrame
# 注意：路径是相对路径，需要确保运行时工作目录或文件位置正确
data = pd.read_csv('./test/house_prices.csv')

# 提取 Price 列（房价）作为一维 Series，用于后续分析
prices = data['Price']

# ==========================================
# 3. 房价分布可视化（1 行 2 列子图布局）
# ==========================================
# plt.subplots(1, 2, figsize=(12, 5)) 创建 1 行 2 列的子图布局
#   1, 2           → 1 行 2 列，共 2 个子图
#   figsize=(12,5) → 整张图的宽高（单位：英寸），宽 12 高 5
#   fig            → Figure 对象（整张画布）
#   ax1, ax2       → 两个 Axes 对象（分别代表左右两个子图）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ------------------------------------------
# 左侧子图：直方图（带核密度估计曲线）
# ------------------------------------------
# sns.histplot() 绘制直方图，展示数据的频率分布
#   prices        → 要可视化的数据（房价 Series）
#   kde=True      → 叠加核密度估计曲线（Kernel Density Estimate）
#                   KDE 是直方图的平滑版本，用连续曲线展示数据分布形状
#   ax=ax1        → 指定画在左侧子图 ax1 上
#   color='skyblue' → 柱子填充颜色为天蓝色
sns.histplot(prices, kde=True, ax=ax1, color='skyblue')

# 设置 X 轴标签为"房价(万元)"
ax1.set_xlabel('房价(万元)')
# 设置 Y 轴标签为"频数"（出现次数）
ax1.set_ylabel('频数')

# ------------------------------------------
# 右侧子图：箱线图（水平方向）
# ------------------------------------------
# ax2.boxplot() 绘制箱线图，用于展示数据的五数概括和异常值
#   prices           → 要可视化的数据
#   vert=False       → 水平方向绘制（默认垂直）
#   widths=0.7       → 箱体宽度为 0.7（占可用空间的比例）
#   patch_artist=True → 允许对箱体进行填充着色（默认 False 只有边框）
#   boxprops=dict(facecolor='lightgreen')
#       → 设置箱体填充颜色为浅绿色
#         boxprops 是传递给箱线图属性的字典
ax2.boxplot(prices, vert=False, widths=0.7, patch_artist=True,
            boxprops=dict(facecolor='lightgreen'))

# 设置 X 轴标签
ax2.set_xlabel('房价(万元)')

# ------------------------------------------
# 添加标题
# ------------------------------------------
# fig.suptitle() 设置整张图的总标题（显示在画布顶部居中）
fig.suptitle('房屋价格分布特征')
# ax1.set_title() 设置左侧子图的独立标题
ax1.set_title('房价直方图')

# ==========================================
# 4. 设置中文字体（解决中文显示为方块的问题）
# ==========================================
# SimHei（黑体）是 Windows 系统自带的中文字体
# 如果不设置此参数，matplotlib 默认字体不支持中文，会显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']
# axes.unicode_minus=False 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 5. 调整布局并显示
# ==========================================
# plt.tight_layout() 自动调整子图间距，防止标签和标题相互重叠
plt.tight_layout()
# plt.show() 将图表渲染到屏幕上显示（阻塞，直到关闭窗口）
plt.show()
