import torch

pthfile = r'checkpoints/model-45.pth' #faster_rcnn_ckpt.pth
net = torch.load(pthfile,map_location=torch.device('cpu')) # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

print(type(net)) # 类型是 dict
print(len(net)) # 长度为 4，即存在四个 key-value 键值对

for k in net.keys():
 print(k) # 查看四个键，分别是 model,optimizer,scheduler,iteration
