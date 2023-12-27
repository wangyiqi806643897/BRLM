import torch

pthfile = r'checkpoints/model-45.pth' #faster_rcnn_ckpt.pth
net = torch.load(pthfile,map_location=torch.device('cpu')) 
print(type(net)) # dict type
print(len(net)) # four key-value

for k in net.keys():
 print(k) # four keys model,optimizer,scheduler,iteration
