import os
import shutil
import time
import json
import pickle
from typing import Dict

import numpy as np
from functools import reduce

import torch

from .metrics import ANETdetection


def load_results_from_pkl(filename):
    # load from pickle file
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def load_results_from_json(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        results = json.load(f)
    # for activity net external classification scores
    if 'results' in results:
        results = results['results']
    return results

def results_to_dict(results):
    """convert result arrays into dict used by json files"""
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = []

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score']
    ):
        results_dict[vidx].append(
            {
                "label" : int(label),
                "score" : float(score),
                "segment": [float(start), float(end)],
            }
        )
        
    return results_dict


def results_to_array(results):
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = {
            'label'   : [],
            'score'   : [],
            'segment' : [],
            # 'temp_idxs': [],
        }

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score'],
    ):
        results_dict[vidx]['label'].append(int(label))
        results_dict[vidx]['score'].append(float(score))
        results_dict[vidx]['segment'].append(
            [float(start), float(end)]
        )

    for vidx in vidxs:
        label = np.asarray(results_dict[vidx]['label'])
        score = np.asarray(results_dict[vidx]['score'])
        segment = np.asarray(results_dict[vidx]['segment'])

        # the score should be already sorted, just for safety
        results_dict[vidx]['label'] = label
        results_dict[vidx]['score'] = score
        results_dict[vidx]['segment'] = segment
    
    return results_dict


def postprocess_results(results, cls_score_file, prior_prob=0.15, num_pred=200, topk=2):
    # load results and convert to dict
    if isinstance(results, str):
        results = load_results_from_pkl(results)
    # array -> dict
    class_names = results['class_names']
    results = results_to_array(results)

    # load external classification scores
    if '.json' in cls_score_file:
        with open(cls_score_file, "rt") as f:
            data = json.load(f)
        cls_scores = data['results']
        full_class_names = data['class']
        class_name_to_idx = {
            name: i
            for i, name in enumerate(full_class_names)
        }
    else:
        cls_scores = load_results_from_pkl(cls_score_file)
    
    # dict for processed results
    processed_results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': [],
    }

    # process each video
    for i, (vid, result) in enumerate(results.items()):
        # pick top k cls scores and idx
        curr_cls_scores = np.asarray(cls_scores[vid])
        class_indices = np.asarray([class_name_to_idx[name] for name in class_names[i]])
        # class_indices = np.asarray([ 0,  2,  3,  9, 10, 11, 14, 16, 17, 18])   # split 0
        # import ipdb; ipdb.set_trace()

        curr_cls_scores = curr_cls_scores[class_indices]
        
        topk = max((curr_cls_scores > prior_prob).sum(), 1)
        topk_cls_indices = np.argsort(curr_cls_scores)[::-1][:topk]
        
        valid_mask = [result['label'] == topk_cls_idx for topk_cls_idx in topk_cls_indices]
        valid_mask = reduce(np.logical_or, valid_mask)
        
        pred_score = result['score'][valid_mask]
        pred_segment = result['segment'][valid_mask]
        pred_label = result['label'][valid_mask]
        inds = np.argsort(pred_score)[::-1][:num_pred]
        pred_score = pred_score[inds]
        pred_segment = pred_segment[inds]
        pred_label = pred_label[inds]
        
        num_segs = min(num_pred, len(pred_score))
        
        processed_results['video-id'].extend([vid] * len(pred_score))
        processed_results['t-start'].append(pred_segment[:, 0])
        processed_results['t-end'].append(pred_segment[:, 1])
        processed_results['label'].append(pred_label)
        processed_results['score'].append(pred_score)

    processed_results['t-start'] = np.concatenate(
        processed_results['t-start'], axis=0)
    processed_results['t-end'] = np.concatenate(
        processed_results['t-end'], axis=0)
    processed_results['label'] = np.concatenate(
        processed_results['label'],axis=0)
    processed_results['score'] = np.concatenate(
        processed_results['score'], axis=0)

    return processed_results
