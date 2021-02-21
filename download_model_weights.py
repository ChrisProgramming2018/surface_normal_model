from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
from models import VisualPrior, VisualPriorRepresentation
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import torch
import time
from collections import deque
import torch.optim as optim



def time_format(sec):
    """
    Args:
    param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)

t0 = time.time()

#vp = VisualPrior()
#feature_tasks= ["normal"]
#vr = VisualPriorRepresentation()
#vr._load_unloaded_nets(feature_tasks)
TASKONOMY_PRETRAINED_WEIGHT_FILES= ["normal_decoder-8f18bfb30ee733039f05ed4a65b4db6f7cc1f8a4b9adb4806838e2bf88e020ec.pth", "normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth"]
#path = "pretrained_model_weights"
path_de = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[0])
path_en = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[1])
model = TaskonomyNetwork(load_encoder_path=path_en, load_decoder_path=path_de)
model.encoder.eval_only = False
model.decoder.eval_only = False

for param in model.parameters():
        param.requires_grad = True


model.train()

lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

from replay_buffer_depth import ReplayBufferDepth
   
size = 256
memory = ReplayBufferDepth((size, size), (size, size, 3), (size, size, 3), 15000, "cuda")
path = "normals_memory10k"
print("Load buffer ...")
memory.load_memory(path)
print("... buffer loaded")


batch_size = 32
scores_window = deque(maxlen=100) 
epochs = 1000
for epoch in range(epochs):
    rgb_batch, depth_batch, normal_batch = memory.sample(batch_size)
    x_recon = model(rgb_batch)
    
    # loss = -torch.mean(torch.sum(normal_batch * torch.log(1e-5 + x_recon) + (1 - normal_batch) * torch.log(1e-5 + 1 - x_recon), dim=1))
    optimizer.zero_grad()
    loss = F.mse_loss(x_recon, normal_batch)
    loss.backward()
    optimizer.step()
    scores_window.append(loss.item())
    mean_loss = np.mean(scores_window)
    text = "Epochs {}  \ {}  loss {:.2f}  ave loss {:.2f}  time {}  \r".format(epoch, epochs, loss, mean_loss, time_format(time.time() - t0))
    print(text)

rgb_batch, depth_batch, normal_batch = memory.sample(batch_size)

#rgb_batch= TF.to_tensor(rgb_batch)
print("r", rgb_batch.shape)
y = model(rgb_batch)
loss = F.mse_loss(rgb_batch, y)
print(loss)
x = rgb_batch
x_recon = y
recon_term = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon) + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
print(x_recon.shape)
sys.exit()
"""
resized = []
for i in y:
    print(i.shape)
    obs = cv2.resize(np.array(i.transpose(2,0,1)), dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    print(obs.shape)
    sys.exit()
    resized.append(TF.to_tensor(obs))

y = torch.stack(resized, dim=0)

print(y.shape)
loss = F.mse_loss(rgb_batch, y)
print(loss)
print(y.shape)
sys.exit()
#rgb_batch= TF.to_tensor(rgb_batch)
# y = model(rgb_batch)
print(normal_batch.shape)
# x = x.transpose(1,2,0)
#red, green, blue = x.T 
#x = np.array([blue, green, red])
#print(x.shape)
#x = x.transpose()
"""

x = memory.obses[0]
print("from memory", x.shape)
#cv2.imshow("depth_image", x.transpose(1,2,0))
#cv2.imshow("depth_image", x[...,::-1])
#cv2.imshow("depth_image", n)
#cv2.imshow("depth_image", x)
#cv2.waitKey(0)

x = TF.to_tensor(x)
#image = Image.open('test.png')
# x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
#x = TF.to_tensor(TF.resize(image, 84)) #* 2 - 1
#x1 = np.array(image) #* 2 - 1
#cv2.imshow("depth_image", x1.transpose(1,2,0))
#cv2.imshow("depth_image", x1)
#cv2.waitKey(0)
print("before stack ", x.shape)
x = torch.stack([x,x], dim=0)
# x = x.unsqueeze_(0)
print("input ", x.shape)
#print(model)
y = model(x)
print(y.shape)
print(y)
print(y.min(), y.max())
TF.to_pil_image(y[0] / 2. + 0.5).save('test_normals_readout.png')
