#!/usr/bin/python3

"""Train module.

Usage:
    python train.py <task> <model_name>

"""



import argparse
from os.path import isfile, join
from typing import List, Dict, Union
import numpy as np
import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4' #7

from scipy import sparse
# 忽略代码中输出的警告
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from config import GlobalConfig
from constant import (TEXT_TASK,)
from constant import TASK_STR
from dataset import Dataset
from lib.train import text_train
from util import load_pkl, get_embed_init, load_pkl2



TASKS: List[str] = list(TASK_STR.values())
base_dir = '/home/ma-user/work/chenxiaolin/dataset_dump/'

knowledge_dir = '/home/ma-user/work/chenxiaolin/knowledge_extract/triplet/'
path_tip_vec = knowledge_dir + 'tip_vec_corresponding.npy'
path_triplet_vec = knowledge_dir + 'triplet_vec_corresponding.npy'
tip_vec = np.load(path_tip_vec).tolist() #tip_vec,triplet_vec
triplet_vec = np.load(path_triplet_vec).tolist()


path_train_context_shuffle = base_dir + 'train_context_shuffle.npy'
path_valid_context_shuffle = base_dir + 'valid_context_shuffle.npy'
path_test_context_shuffle = base_dir + 'test_context_shuffle.npy'

train_context_index = np.load(path_train_context_shuffle).tolist()  #train_context_index,valid_context_index, test_context_index
valid_context_index = np.load(path_valid_context_shuffle).tolist()
test_context_index = np.load(path_test_context_shuffle).tolist()

train_context_dir = base_dir + 'Context/train/'
valid_context_dir = base_dir + 'Context/validate/' # valid_context_dir, test_context_dir
test_context_dir = base_dir + 'Context/test/'



def train(task: int, model_file_name: str):
    """Train model.

    Args:
        task (int): Task.
        model_file_name (str): Model file name (saved or to be saved).

    """

    path_dialog_vocab = base_dir + 'dialog_vocab.npy'
    path_glove = base_dir + 'glove.npy'

    dialog_vocab = np.load(path_dialog_vocab, allow_pickle=True).tolist()
    glove = np.load(path_glove, allow_pickle=True).tolist()
    print('vocab_glove loaded!')

    # Dataset wrap.
    # dialog_vacb; global_repre
    train_dataset = Dataset(task, train_context_dir, dialog_vocab,  train_context_index)
    valid_dataset = Dataset(task, valid_context_dir, dialog_vocab, valid_context_index)
    test_dataset = Dataset(task, test_context_dir, dialog_vocab,  test_context_index)

    
    vocab_size = 4892
    embed_init = get_embed_init(
        glove, vocab_size)
    # 4892, 300
    embed_init = embed_init.to(GlobalConfig.device)  


    model_dump_dir = '/cache/chenxiaolin/model_file/'
    # Load model file.
    # model_file = join(DatasetConfig.dump_dir, model_file_name)
    model_file = join(model_dump_dir, model_file_name)
    # Task-specific parts---text/image
    if task == TEXT_TASK:
        print('----text_train---------')
        text_train(
                    train_dataset,
                    valid_dataset,
                    test_dataset,                    
                    model_file,
                    dialog_vocab,
                    embed_init
                )
 

def parse_cmd() -> Dict[str, List[str]]:
    """Parse commandline parameters.

    Returns:
        Dict[str, List[str]]: Parse result.

    """

    # Definition of argument parser.
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument(
        'task',
        metavar='<task>',
        choices=TASKS,
        default='text',
        help='task ({})'.format('/'.join(TASKS))
    )
    parser.add_argument('model_file', metavar='<model_file>',default='model')

    # Namespace -> Dict
    parse_res: Dict[str, Union[List[str], str]] = vars(parser.parse_args())
    return parse_res


def main():

    model_file = 'model_244_7'

    task: int = TEXT_TASK
    # model_file: str = parse_result['model_file']
    train(task, model_file)

if __name__ == '__main__':
    torch.manual_seed(777777)
    torch.cuda.manual_seed_all(777777)
    main()
