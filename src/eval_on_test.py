
import argparse
import json
import logging

# import torch
import os

from algorithm.policies import ModelOptions
from captioning import eval_utils
from captioning.dataloader import DataLoader
# from captioning.dataloaderraw import DataLoaderRaw
from captioning.experiment import CaptionOptions
from captioning.nets import FCModel


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_nicnes_path', type=str, help='path to pth model',
                        default='./logs/logs/nic_nes_mscoco_fc_caption_2557/models/best/best_elite/0_0_elite.pth')

    parser.add_argument('--model_nices_path', type=str, help='path to pth model',
                        default='./logs/logs/nic_es_mscoco_fc_caption_13411/best/best_elite/0_0_elite.pth')

    parser.add_argument('--model_xent_path', type=str, help='path to pth model',
                        default='../instance_marijke_gpu/logs/xent/model-best.pth')

    parser.add_argument('--model_sc_path', type=str, help='path to pth model',
                        default='../instance_marijke_gpu/logs/sc_bu_c2/model-best.pth')

    parser.add_argument('--input_json', type=str, default='data/cocotalk.json', help='')
    parser.add_argument('--from_dir', type=str, default='')
    parser.add_argument('--from_karp_testset', type=bool, default=True)
    parser.add_argument('--from_mscoco_testset', type=bool, default=False)

    args = parser.parse_args()

    with open(args.input_json) as inp:
        info = json.load(inp)
    ix_to_word = info['ix_to_word']

    model_options = ModelOptions(**{
        'rnn_size': 128,
        'vocab_size': len(ix_to_word),
        'input_encoding_size': 128,
        'fc_feat_size': 2048,
    })

    model_paths = {
        'nicnes': args.model_nicnes_path,
        'nices': args.model_nices_path,
        'xent': args.model_xent_path,
        'sc': args.model_sc_path,
    }
    models = {
        name: FCModel(from_param_file=path, options=model_options).eval()
        for (name, path) in model_paths.items() if path
    }

    # if args.from_dir:
    #     loader = DataLoaderRaw({'folder_path': args.from_dir,
    #                             'coco_json': '',
    #                             'batch_size': 32,
    #                             'cnn_model': 'resnet101'})

    if args.from_karp_testset:
        dl_options = CaptionOptions(**{
            'input_json': 'data/cocotalk.json', 
            'input_fc_dir': 'data/cocobu_fc',
            'input_label_h5': 'data/cocotalk_label.h5'
        })
        # config = Config(**{})
        loader = DataLoader(dl_options, config=None, batch_size=32)

    else:  # args.from_mscoco_testset:
        dl_options = CaptionOptions(**{
            'input_json': 'data/cocotalk.json',
            # todo change
            'input_fc_dir': 'data/cocobu_fc',
            'input_label_h5': 'data/cocotalk_label.h5'
        })
        loader = DataLoader(dl_options, config=None, batch_size=32)

    all_stats = {}
    preds_per_model = {}
    for (name, model) in models.items():
        stats, predictions = eval_utils.eval_split(model, loader, './output', do_eval=True,
                                                   split='test', num=5000, incl_gts=True)
        if stats is not None:
            logging.info('%s: %s', name, stats)

        all_stats[name] = stats
        preds_per_model[name] = predictions

    preds_per_img = {}
    for name, preds in preds_per_model.items():
        for entry in preds:
            if entry['image_id'] in preds_per_img:
                tmp = preds_per_img[entry['image_id']]
            else:
                tmp = {'gts': entry['gts']}
            tmp[name] = entry['caption']
            preds_per_img[entry['image_id']] = tmp

    all_output = {
        'stats': all_stats,
        'preds_per_img': preds_per_img,
        'preds_per_model': preds_per_model,
    }
    with open('output/test_output_{}.json'.format(os.getpid()), 'w') as f:
        # entry['image_id'], entry['caption']
        json.dump(all_output, f)


def inspect_captions(output_file, idx1, n):
    import subprocess
    from algorithm.tools.utils import find_file_with_pattern
    with open(output_file, 'rb') as f:
        all_captions = json.load(f)

    directory = '/Users/rubencartuyvels/Documents/bir-18-19/thesis/ga-img-captioning/data'
    i, count = idx1, 0
    while count < n:
        imgid = list(all_captions['preds_per_img'].keys())[i]
        captions_for_img = all_captions['preds_per_img'][imgid]
        generated_capts = [v for (k, v) in captions_for_img.items() if k != 'gts']
        i += 1

        # only print examples for which all captions are different
        if len(set(generated_capts)) == len(generated_capts):
            filename = 'COCO_val2014_000000{}.jpg'.format(
                imgid if len(str(imgid)) == 6
                else '0' * (6 - len(str(imgid))) + str(imgid))
            count += 1
            print(imgid)
            print(filename)
            if not find_file_with_pattern(filename, directory):
                print('http://cocodataset.org/#explore?id={}'.format(imgid))
                subprocess.call([
                    'bash', '-c',
                    'gcloud compute scp --recurse instance-1:/home/rubencartuyvels/ga-img-captioning/data/val2014/{} '
                    '{}/'.format(filename, directory)])
            print(captions_for_img)


if __name__ == '__main__':
    # Inspect captions:
    # inspect_captions('../output/test_output_95932.json', 100, 20)

    # Run test evaluation:
    # python -u eval_on_test.py

    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )
    run()
