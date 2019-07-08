"""
    Code from https://github.com/ruotianluo/self-critical.pytorch
"""

import torch
import numpy as np
import json
import os


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
    sys.path.append('cococaption')
    ann_file = 'cococaption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    cache_path = os.path.join(directory, 'eval_cache_' + split + '.json')

    coco = COCO(ann_file)
    valids = coco.getImgIds()

    preds_filt = [p for p in preds if p['image_id'] in valids]

    with open(cache_path, 'w') as f:
        json.dump(preds_filt, f)  # serialize to temporary json file. Sigh, COCO API...

    coco_res = coco.loadRes(cache_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    coco_eval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in coco_eval.eval.items():
        out[metric] = score

    return out


def eval_split(model, loader, directory, num=-1, split='val'):
    model.eval()

    # Todo we assume model is on single device
    device = next(model.parameters()).device

    loader.reset_iterator(split)

    n = 0
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # Only leave one feature for each image, in case duplicate sample
        fc_feats = torch \
            .from_numpy(data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img]) \
            .to(device)

        # forward the model to also get generated samples for each image
        seq = model(fc_feats, greedy=True)[0].data

        sents = decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)

        ix1 = data['bounds']['it_max']
        if num != -1:
            ix1 = min(ix1, num)
        for i in range(n - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break

        if num >= 0 and n >= num:
            break

    lang_stats = language_eval(predictions, directory, split)
    return lang_stats
