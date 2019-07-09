
import argparse
import json
import logging

# import torch
from algorithm.policies import ModelOptions
from captioning import eval_utils
from captioning.dataloader import DataLoader
# from captioning.dataloaderraw import DataLoaderRaw
from captioning.experiment import CaptionOptions
from captioning.nets import FCModel


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_nicnes_path', type=str, help='path to pth model',
                        default='./logs/logs/_es_mscoco_fc_caption_5741/best/best_elite/0_0_elite.pth')

    parser.add_argument('--model_nices_path', type=str, help='path to pth model',
                        default='./logs/logs/_ga_mscoco_fc_caption_59458/best_elite/0_0_elite.pth')

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
                                                   split='test', num=5000, verbose=False)  # todo num?
        if stats is not None:
            logging.info('%s: %s', name, stats)

        all_stats[name] = stats
        preds_per_model[name] = predictions

    preds_per_img = {}
    for name, preds in preds_per_model.items():
        for entry in preds:
            tmp = preds_per_img[entry['image_id']] if entry['image_id'] in preds_per_img else {}
            tmp[name] = entry['caption']
            preds_per_img[entry['image_id']] = tmp

    all_output = {
        'stats': all_stats,
        'preds_per_img': preds_per_img,
        'preds_per_model': preds_per_model,
    }
    with open('output/test_output.json', 'w') as f:
        # entry['image_id'], entry['caption']
        json.dump(all_output, f)


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )
    run()
