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
