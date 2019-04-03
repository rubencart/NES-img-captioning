"""
Code from https://github.com/ruotianluo/self-critical.pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import logging

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys

# import misc.utils as utils
import algorithm.tools.utils as utils


# Input: seq, N*D numpy array, with elements 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out


def language_eval(preds, directory, split):
    import sys
    sys.path.append("cococaption")
    annFile = 'cococaption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    # if not os.path.isdir('eval_results'):
    #     os.mkdir('eval_results')
    # cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    # if not os.path.isdir('logs/eval_results'):
    #     os.makedirs('logs/eval_results')

    # utils.mkdir_p('logs/eval_results')
    # cache_path = os.path.join('logs/eval_results/', model_id + '_' + split + '.json')
    cache_path = os.path.join(directory, 'eval_cache_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]

    logging.info('using %d/%d predictions' % (len(preds_filt), len(preds)))

    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    print('json dumped')

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    print('evaluated')

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    print('out constructed')

    # imgToEval = cocoEval.imgToEval
    # for p in preds_filt:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption
    # with open(cache_path, 'w') as outfile:
    #     json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    logging.info('   *** OUT : {} ****    '.format(out))
    return out


def eval_split(model, loader, directory, num=-1, split='val'):
    print('EVAL SPLIT IS CALLED')
    # verbose = eval_kwargs.get('verbose', False)
    # verbose_beam = eval_kwargs.get('verbose_beam', 1)
    # verbose_loss = eval_kwargs.get('verbose_loss', 1)
    # num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    # split = eval_kwargs.get('split', 'val')
    # lang_eval = eval_kwargs.get('language_eval', 0)
    # dataset = eval_kwargs.get('dataset', 'coco')
    # beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    # Todo we assume model is on single device
    device = next(model.parameters()).device

    loader.reset_iterator(split)
    # loader.reset()

    n = 0
    # loss = 0
    # loss_sum = 0
    # loss_evals = 1e-8
    predictions = []
    while True:
    # for (n, data) in enumerate(loader):
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # if data.get('labels', None) is not None and verbose_loss:
        #     # forward the model to get loss
        #     tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        #     # tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        #     tmp = [torch.from_numpy(_).to(device) if _ is not None else _ for _ in tmp]
        #     fc_feats, att_feats, labels, masks, att_masks = tmp
        #
        #     with torch.no_grad():
        #         loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:]).item()
        #     loss_sum = loss_sum + loss
        #     loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [
            data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],

            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]
            if data['att_masks'] is not None else None
        ]

        # tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        tmp = [torch.from_numpy(_).to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            # seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
            # todo opt ok to leave?
            seq = model(fc_feats, att_feats, att_masks, opt={}, mode='sample')[0].data

        # Print beam search
        # if beam_size > 1 and verbose_beam:
        #     for i in range(loader.batch_size):
        #         print('\n'.join(
        #             [decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
        #         print('--' * 10)
        sents = decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            # if eval_kwargs.get('dump_path', 0) == 1:
            #     entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            # todo could be interesting --> see images + captions!
            # if eval_kwargs.get('dump_images', 0) == 1:
            #     # dump the raw image to vis/ folder
            #     cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
            #                                 data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
            #         len(predictions)) + '.jpg'  # bit gross
            #     print(cmd)
            #     os.system(cmd)

            # if verbose:
            #     print('image %s: %s' % (entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        # EDIT: not possible anymore
        # ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num != -1:
            ix1 = min(ix1, num)
        for i in range(n - ix1):
            predictions.pop()

        # if verbose:
        #     print('evaluating validation performance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break

        if num >= 0 and n >= num:
            break

    print('OUT OF WHILE LOOP')
    # lang_stats = None
    # if lang_eval == 1:
        # lang_stats = language_eval(predictions, eval_kwargs['id'], split)
    lang_stats = language_eval(predictions, directory, split)

    # Switch back to training mode
    # model.train()
    return lang_stats
