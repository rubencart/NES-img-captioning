# coding: utf-8
# coding: utf-8
import torch
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
    sens = torch.load(f)
    
len(sens)
2048 * 128 + 128
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640

slice_stats(0, 2048 * 128)
def slice_stats(start, nb):
    slc = sens[start:start+nb]
    return slc.min(), slc.median(), slc.mean(), slc.max()
    
slice_stats(0, 2048 * 128)
slice_stats(2048 * 128, 128)
slice_stats(2048 * 128 + 128, 9488 * 128)
# last was embed weight
# now comes logit weight
slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)
# logit bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128, 9488)
# i2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488, 640 * 128)
# i2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128, 640)
# h2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640, 128 * 640)
# h2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 128 * 640, 640)
slice_stats(-640)
slice_stats(-640, 0)
slice_stats(-640, 640)
slc = sens[-640:]
slc.min(), slc.median(), slc.mean(), slc.max()
torch.equal(sens[-640:], sens[])
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640
torch.equal(sens[-640:], sens[2782608:2783248])
# coding: utf-8
import torch
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
    sens = torch.load(f)
    
len(sens)
2048 * 128 + 128
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640

slice_stats(0, 2048 * 128)
def slice_stats(start, nb):
    slc = sens[start:start+nb]
    return slc.min(), slc.median(), slc.mean(), slc.max()
    
slice_stats(0, 2048 * 128)
slice_stats(2048 * 128, 128)
slice_stats(2048 * 128 + 128, 9488 * 128)
# last was embed weight
# now comes logit weight
slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)
# logit bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128, 9488)
# i2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488, 640 * 128)
# i2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128, 640)
# h2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640, 128 * 640)
# h2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 128 * 640, 640)
slice_stats(-640)
slice_stats(-640, 0)
slice_stats(-640, 640)
slc = sens[-640:]
slc.min(), slc.median(), slc.mean(), slc.max()
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640
torch.equal(sens[-640:], sens[2782608:2783248])
# coding: utf-8
import torch
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
    sens = torch.load(f)
    
len(sens)
2048 * 128 + 128
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640

def slice_stats(start, nb):
    slc = sens[start:start+nb]
    return slc.min(), slc.median(), slc.mean(), slc.max()
    
slice_stats(0, 2048 * 128)
slice_stats(2048 * 128, 128)
slice_stats(2048 * 128 + 128, 9488 * 128)
# last was embed weight
# now comes logit weight
slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)
# logit bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128, 9488)
# i2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488, 640 * 128)
# i2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128, 640)
# h2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640, 128 * 640)
# h2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 128 * 640, 640)
slice_stats(-640)
slice_stats(-640, 0)
slice_stats(-640, 640)
slc = sens[-640:]
slc.min(), slc.median(), slc.mean(), slc.max()
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640
torch.equal(sens[-640:], sens[2782608:2783248])
# coding: utf-8
import torch
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
    sens = torch.load(f)
    
len(sens)
2048 * 128 + 128
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640

def slice_stats(start, nb):
    slc = sens[start:start+nb]
    return slc.min(), slc.median(), slc.mean(), slc.max()
    
slice_stats(0, 2048 * 128)
slice_stats(2048 * 128, 128)
slice_stats(2048 * 128 + 128, 9488 * 128)
# last was embed weight
# now comes logit weight
slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)
# logit bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128, 9488)
# i2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488, 640 * 128)
# i2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128, 640)
# h2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640, 128 * 640)
# h2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 128 * 640, 640)
slice_stats(-640, 0)
slice_stats(-640, 640)
slc = sens[-640:]
slc.min(), slc.median(), slc.mean(), slc.max()
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640
torch.equal(sens[-640:], sens[2782608:2783248])
# coding: utf-8
import torch
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
    sens = torch.load(f)
    
len(sens)
2048 * 128 + 128
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640
2048 * 128 + 128 + 9487 * 128 + 128 + 9487 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9487 + 640 * 128 + 640 + 640 * 128 + 640
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640

def slice_stats(start, nb):
    slc = sens[start:start+nb]
    return slc.min(), slc.median(), slc.mean(), slc.max()
    
slice_stats(0, 2048 * 128)
slice_stats(2048 * 128, 128)
slice_stats(2048 * 128 + 128, 9488 * 128)
# last was embed weight
# now comes logit weight
slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)
# logit bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128, 9488)
# i2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488, 640 * 128)
# i2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128, 640)
# h2h weight
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640, 128 * 640)
# h2h bias
slice_stats(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 128 * 640, 640)
slc = sens[-640:]
slc.min(), slc.median(), slc.mean(), slc.max()
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640
torch.equal(sens[-640:], sens[2782608:2783248])
sens = torch.zeros(2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640)
2048 * 128 + 128 + 9488 * 128 + 9488 * 128 + 9488 + 640 * 128 + 640 + 640 * 128 + 640
slice_stats(0, 2048 * 128)
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
   sens = torch.load(f)
   
t = torch.zeros(2865808)
slice_stats(0, 2048 * 128)
t[:2048 * 128] = 33.2562
slice_stats(2048 * 128, 128)
t[2048 * 128:128] = 6.8203
t[2048 * 128 + 128 : 9488 * 128] = slice_stats(2048 * 128 + 128, 9488 * 128)[3].item()
t[2048 * 128 + 128 + 5]
t[2048 * 128 + 128 + 6]
t[2048 * 128 + 128 + 9488 * 128: 9488 * 128] = slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)
t[2048 * 128 + 128 + 9488 * 128: 9488 * 128] = slice_stats(2048 * 128 + 128 + 9488 * 128, 9488 * 128)[3].item()
breakpoints = [0, 2048*128, 128, 9488*128, 9488 * 128, 9488, 640 * 128, 640, 640 * 128, 640]
len(t)
t
count = 0
for i in range breakpoints[1:]:
    t[count:count+i] = slice_stats(count, count+i)[3].item()
for i in breakpoints[1:]:
    t[count:count+i] = slice_stats(count, count+i)[3].item()
    count += i
    
from matplotlib import pyplot as plt
plt.plot(np.arange(len(t.numpy())), t.numpy())
import numpy as np
plt.plot(np.arange(len(t.numpy())), t.numpy())
plt.savefig('logs/sens-t.pdf', format='pdf')
for i in breakpoints[1:]:
    t[count:count+i] = sens[count:count+i].max()
    count += i
print(count)
sens
with open('logs/ga_mscoco_fc_caption_37717/models/offspring/sens_t0_p0.txt', 'rb') as f:
   sens = torch.load(f)
   
sens
for i in breakpoints[1:]:
    t[count:count+i] = sens[count:count+i].max()
    count += i
print(count)
breakpoints
breakpoints = breakpoints[1:]
breakpoints
count = 0
for i in breakpoints:
    t[count:count+i] = sens[count:count+i].max()
    count += i
print(count)
count = 0
for i in breakpoints:
    print(sens[count:count+i].max())
    t[count:count+i] = sens[count:count+i].max()
    count += i
print(count)
plt.close(fig)
plt.plot(np.arange(len(t.numpy())), t.numpy())
plt.savefig('logs/sens-t.pdf', format='pdf')
plt.close()
plt.plot(np.arange(len(t.numpy())), t.numpy())
plt.savefig('logs/sens-t.pdf', format='pdf')
t.min()
t /= t.min()
t.min()
t.max()
torch.save(t, 'logs/sens_test.pt')
torch.save(t, 'logs/sens_test.pth')
