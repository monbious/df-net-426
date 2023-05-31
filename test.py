import torch

# 创建示例张量
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# 定义维度大小
dim1 = 3
dim2 = 3

# 创建多维索引
indices = torch.meshgrid(
    torch.arange(dim1),
    torch.arange(dim2)
)
print(indices)

# 将多维索引拼接为一个张量
tensor_indices = torch.stack(indices, dim=-1)
print(tensor_indices)

# 根据索引获取值
values = tensor[tensor_indices[:, :, 0], tensor_indices[:, :, 1]]

print(values)
