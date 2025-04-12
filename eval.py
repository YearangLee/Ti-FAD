# python imports
import argparse
import glob
import os
import time
from pprint import pprint

# torch imports
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed, valid_one_epoch, ModelEma
from libs.modeling.text import data_split_dir

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    # assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], args.n, False, cfg['val_split'], **cfg['dataset']
    )

    label_dict = val_dataset.label_dict
    label_to_name = {
        v: k
        for k, v in label_dict.items()
    }
    
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(args.n, 
                           cfg['model_name'], 
                           cfg['dataset']['subset_file'], 
                           cfg['dataset']['data_split'], 
                           **cfg['model']
                           )
   
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # model_ema = ModelEma(model)

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )

    model.load_state_dict(checkpoint['state_dict'], strict=True)  
    del checkpoint      

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        subset_file = data_split_dir(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], mode='test', split_num=args.n)
        

        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds'],
            subset_file=subset_file,    
        )

    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    if 'ema' not in args.ckpt:
        mAP, tiou_mAP_list = valid_one_epoch(
            val_loader,
            model,
            -1,
            evaluator=det_eval,
            output_file=None,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=args.print_freq,
            prior_prob=cfg['opt']['prior_prob'],
            num_pred=cfg['opt']['num_pred']
        )

    end = time.time()
    print(tiou_mAP_list)
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--n', default=0, type=int,
                        help='split number')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()

    main(args)
