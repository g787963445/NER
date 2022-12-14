import torch
x = torch.Tensor(3,1,4,1,2)
print(x.shape)
a = x.squeeze(dim=1)  # 成员函数删除第二维度
print(a.shape)
b = torch.squeeze(x, dim=1)  # torch自带函数删除第二维度
print(b.shape)
c = torch.squeeze(x, dim=3)  # 删除第三维度
print(c.shape)
d = torch.squeeze(x)  # 若不标注删除第几维度，则会删除所有为1的维度
print(d.shape)
e = torch.unsqueeze(x,dim=5)
print(e.shape)
