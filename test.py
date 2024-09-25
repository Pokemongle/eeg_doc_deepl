avg_heatmap_1 /= count_0  # 求平均
avg_heatmap_1 = F.relu(avg_heatmap_1)  # 负值归零
avg_heatmap_1 /= torch.max(avg_heatmap_1)  # 归一化到[0, 1]
avg_heatmap_1 = avg_heatmap_1.cpu().detach().numpy()
avg_heatmap_1 = np.resize(avg_heatmap_1, (10, 11))
avg_heatmap_1 = avg_heatmap_1[:9, :11]# 只选择前9行和前11列