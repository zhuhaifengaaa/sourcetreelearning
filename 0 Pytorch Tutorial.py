import torch
x = torch.randn(4, 5)
y = torch.randn(4, 5)
z = torch.randn(4, 5)
print('x',x)
print('y',y)
m = torch.max(x)
# print(m)
m, idx = torch.max(input=x, dim=0)
# print('m:', m, 'idx:', idx)

m, idx = torch.max(x, 0, False)
print('1',m, idx)
p=(m,idx)
torch.max(x, 0, False, out=p)
print('2',p[0],p[1])
t = torch.max(x,y)
print('t',t)



