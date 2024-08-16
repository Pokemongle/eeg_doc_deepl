import numpy as np
import matplotlib.pyplot as plt


def draw_eegmap(electric_nodes, matrix_index):
    # 创建图像
    fig, axs = plt.subplots(1, 2, figsize=(20, 11))

    # 绘制第一个图：显示位置对应的数字
    ax1 = axs[0]

    for i in range(matrix_index.shape[1]):
        x = matrix_index[1, i]
        y = -matrix_index[0, i]
        ax1.text(x, y, str(i + 1), ha='center', va='center', fontsize=20,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-5, 6)
    ax1.set_xticks(range(-1, 12))
    ax1.set_yticks(range(-5, 6))
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Position Numbers')

    # 绘制第二个图：显示位置对应的电极标签
    ax2 = axs[1]

    for i, node in enumerate(electric_nodes):
        x = matrix_index[1, i]
        y = -matrix_index[0, i]
        ax2.text(x, y, node, ha='center', va='center', fontsize=20,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
    ax2.set_xlim(-1, 12)
    ax2.set_ylim(-5, 6)
    ax2.set_xticks(range(-1, 12))
    ax2.set_yticks(range(-5, 6))
    ax2.grid(True)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Electrode Labels')

    # 调整子图位置使其居中
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3)

    # 显示图像
    plt.show()
