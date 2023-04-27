import numpy as np
import matplotlib.pyplot as plt

shops = ["Navigation", "Weather", "Schedule"]
sales_product_1 = [8.8, 11.0, 42.7]
sales_product_2 = [10.5, 33.9, 45.8]

# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(shops))

fig, ax = plt.subplots(figsize=(6, 4))
# 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
ax.bar(xticks, sales_product_1, width=0.25, label="DF-Net", color="moccasin")
# 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
ax.bar(xticks + 0.27, sales_product_2, width=0.25, label="Ours", color="tomato")

for a,b in zip(xticks,sales_product_1):   #柱子上的数字显示
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=7)
 for a, b in zip(xticks+0.27, sales_product_2):  # 柱子上的数字显示
     plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=7)

# ax.set_title("Zero-shot Performance", fontsize=15)
# ax.set_xlabel("Domains")
ax.set_ylabel("F1 score (%)")
plt.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3, axis='y')
ax.legend(loc='upper left')

# 最后调整x轴标签的位置
ax.set_xticks(xticks + 0.135)
ax.set_xticklabels(shops)
plt.show()