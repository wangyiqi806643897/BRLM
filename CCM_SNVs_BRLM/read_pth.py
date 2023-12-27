import torch

pthfile = r'checkpoints/model-45.pth' #faster_rcnn_ckpt.pth
net = torch.load(pthfile,map_location=torch.device('cpu')) # Running on CPU

print(type(net)) # dict type
print(len(net)) #  4 key-value

for k in net.keys():
 print(k) # 4 keys: model,optimizer,scheduler,iteration
