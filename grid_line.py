import numpy as np
import matplotlib.pyplot as plt

x_scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
y_scale = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

y_truth = np.array([0.61, 0.67, 0.74, 0.744, 0.78, 0.79])
y_truth_ = np.array([0.52, 0.59, 0.63, 0.66, 0.68, 0.7])

plt.title("Schedule")
plt.xlabel("Schedule Domain")
plt.ylabel("F1 score")

plt.plot(x_scale, y_truth, color='tomato', label='Ours', marker='.', markeredgecolor='tomato',
         markersize='5', markeredgewidth=3)
plt.plot(x_scale, y_truth_, color='moccasin', label='DF-Net', marker='^', markeredgecolor='moccasin',
         markersize='5', markeredgewidth=3)

xticks_labels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
yticks_labels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.xticks(x_scale, xticks_labels,)
plt.yticks(y_scale, yticks_labels,)
plt.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3,)
plt.legend(loc='upper left')
plt.show()



# x_scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# y_scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
#
# y_truth = np.array([0.43, 0.48, 0.59, 0.61, 0.63, 0.62])
# y_truth_ = np.array([0.43, 0.47, 0.56, 0.59, 0.57, 0.555])
#
# plt.title("Weather")
# plt.xlabel("Weather Domain")
# plt.ylabel("F1 score")
#
# plt.plot(x_scale, y_truth, color='tomato', label='Ours', marker='.', markeredgecolor='tomato',
#          markersize='5', markeredgewidth=3)
# plt.plot(x_scale, y_truth_, color='moccasin', label='DF-Net', marker='^', markeredgecolor='moccasin',
#          markersize='5', markeredgewidth=3)
#
# xticks_labels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
# yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# plt.xticks(x_scale, xticks_labels,)
# plt.yticks(y_scale, yticks_labels,)
# plt.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3,)
# plt.legend(loc='upper left')
# plt.show()



# x_scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# y_scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
#
# y_truth = np.array([0.2, 0.25, 0.345, 0.53, 0.58, 0.64])
# y_truth_ = np.array([0.16, 0.22, 0.3, 0.39, 0.46, 0.53])
#
# plt.title("Navigation")
# plt.xlabel("Navigate Domain")
# plt.ylabel("F1 score")
#
# plt.plot(x_scale, y_truth, color='tomato', label='Ours', marker='.', markeredgecolor='tomato',
#          markersize='5', markeredgewidth=3)
# plt.plot(x_scale, y_truth_, color='moccasin', label='DF-Net', marker='^', markeredgecolor='moccasin',
#          markersize='5', markeredgewidth=3)
#
# xticks_labels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
# yticks_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# plt.xticks(x_scale, xticks_labels,)
# plt.yticks(y_scale, yticks_labels,)
# plt.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3,)
# plt.legend(loc='upper left')
# plt.show()