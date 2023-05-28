import torch

# 创建一个形状为(3, 4)的张量
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

# 创建一个形状相同的掩码张量
mask = torch.tensor([[True],
                     [False],
                     [True]])

# 使用masked_fill方法进行填充
filled_x = x.masked_fill(mask, 0)

print(filled_x)

print(torch.softmax(torch.full((1, 3), -1e9), dim=-1))
