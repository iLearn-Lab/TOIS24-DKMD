from typing import List, Dict

import torch
from torch.utils.data import DataLoader
import numpy as np
from config import GlobalConfig, TextTestConfig
from dataset import Dataset
from lib import encode_context
from lib.eval import text_eval
# from model import TextEncoder, ImageEncoder, ContextEncoder
from model import TextEncoder, ContextEncoder, HyperConv, UtterFusion, domainHyper
from model import ToHidden, TextDecoder
import scipy.sparse as sp
import random
import warnings
import copy
import subprocess
import sys
from constant import SOS_ID, EOS_ID, PAD_ID

warnings.filterwarnings('ignore')

def target_id2word(target, id2word):
    batch_target = []
    for i in range(target.size(0)):
        each_target = []
        for j in range(2):  # 前2个是context，第3个是target
            add_sep = False
            each_utter_id = target[i][j]
            for each_id in each_utter_id[:20]:  # 每条utter最长是20
                if each_id != PAD_ID and each_id != EOS_ID:
                    each_word = id2word[each_id]
                    if isinstance(each_word, str):
                        each_target.append(each_word.replace("'", ""))
                        add_sep = True
                    else:
                        print(each_word)
                        print(type(each_word))
                        exit()
            if add_sep:
                each_target.append('<SEP>')
        if len(each_target) > 30:  # max_len-30
            each_target[:29].append('<SEP>')
        each_target_1 = " ".join(each_target)  # .replace(' .','.')
        batch_target.append(each_target_1)

    return batch_target


# def context_know(context, knowledge):
#     num_len = 0
#     context_know_incor = []
#     batch_size = len(context)
#     for i in range(batch_size):
#         each_context_know = []
#         each_context = context[i]
#         each_know = knowledge[i]
#         # know_len
#         each_len = len(each_know.split(' '))
#         if each_len > 70:  # knowledge max_len=100
#             each_know = " ".join(each_know.split(' ')[:70])
#         each_context_know.append(each_context + each_know)
#         each_context_know = " ".join(each_context_know).replace(' .', '.')
#         each_context_know = each_context_know.replace(' ,', ',')  # ?
#         each_context_know = each_context_know.replace(' ?', '?')  #
#         each_context_know = each_context_know.replace(' !', '!')  #
#         context_know_incor.append(each_context_know)
#
#     return context_know_incor


def context_know(context, knowledge, context_visual_know):
    context_know_incor = []
    text_visual_know = []
    batch_size = len(context)
    for i in range(batch_size):
        each_know_trans = []
        each_context_know = []
        each_both_know = []
        each_context = context[i]
        each_visual_know = context_visual_know[i]
        each_know = knowledge[i]
        for each_img_know in each_visual_know:
            if each_img_know != '<pad>' and each_img_know != each_know:
                each_know_trans.append(each_img_know)
        if len(each_know_trans) > 0:
            context_visual_know2 = " ".join(each_know_trans)
            if len(context_visual_know2.split(' ')) > 70:
                context_visual_know2 = " ".join(context_visual_know2.split(' ')[:70]) # 只保留前70个
        else:
            context_visual_know2 = '<pad>'
        # know_len
        each_len = len(each_know.split(' '))
        if each_len > 70:  # knowledge max_len=100
            each_know = " ".join(each_know.split(' ')[:70])
        each_context_know.append(each_context + ' '+each_know+' '+context_visual_know2)
        each_context_know = " ".join(each_context_know).replace(' .', '.')
        each_context_know = each_context_know.replace(' ,', ',')  # ?
        each_context_know = each_context_know.replace(' ?', '?')  #
        each_context_know = each_context_know.replace(' !', '!')  #
        context_know_incor.append(each_context_know)

        each_both_know.append(each_know+' '+context_visual_know2)
        each_both_know = " ".join(each_both_know).replace(' .', '.')
        each_both_know = each_both_know.replace(' ,', ',')  # ?
        each_both_know = each_both_know.replace(' ?', '?')  #
        each_both_know = each_both_know.replace(' !', '!')  #
        text_visual_know.append(each_both_know)

    return context_know_incor, text_visual_know


def visual_know(knowledge):
    context_know_incor = []
    batch_size = len(knowledge[0])
    img_size = len(knowledge)
    for i in range(batch_size):
        each_context_know = []
        for j in range(img_size):
            each_visual_spec = knowledge[j][i]  # each_context
        # for each_visual_spec in context_visual_know:  # 视觉信息包含多条
            # know_len
            each_len = len(each_visual_spec.split(' '))
            if each_len > 35:  # knowledge max_len=100
                each_visual_spec = " ".join(each_visual_spec.split(' ')[:35])
            each_context_know.append(each_visual_spec)
        context_know_incor.append(each_context_know)
    return context_know_incor


def text_test(
        to_hidden: ToHidden,
        text_decoder: TextDecoder,
        test_dataset: Dataset,
        text_length: int,
        vocab: Dict[str, int]):
    """Text test.

    Args:
        test_context_index:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        to_hidden (ToHidden): Context to hidden.
        text_decoder (TextDecoder): Text decoder.
        test_dataset (Dataset): Valid dataset.
        text_length (int): Text length.
        vocab (Dict[str, int]): Vocabulary.

    """
    # 4892
    # id2word: List[str] = [None] * len(vocab)
    id2word: List[str] = [None] * 4892
    index = 0
    for word in vocab.keys():
        wid = vocab[word]
        id2word[wid] = word
        index = index + 1
        if index == 4892:
            break

    # for word, wid in vocab.items():
    #     id2word[wid] = word
    context_len_test = 6959

    # Test dataset loader.
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=TextTestConfig.batch_size,
        num_workers=0
        # num_workers=TextTestConfig.num_data_loader_workers
    )

    to_hidden.eval()
    text_decoder.eval()

    output_file = open('text_177_test.out', 'w', encoding='UTF-8')
    output_file_test_name = 'text_177_test.out'


    with torch.no_grad():
        for batch_id, test_data in enumerate(test_data_loader):
            texts, image_features, text_lengths, batch_index, text_venues, image_len1, visual_know_list = test_data
            continue

            context = target_id2word(texts, id2word)
            # context2 = context_know(context, text_venues[0])
            context_visual_know = visual_know(visual_know_list)

            context2, text_visual_know = context_know(context, text_venues[0], context_visual_know)  # origin

            # context2 = context_know(context, text_venues[0], context_visual_know) # origin

            image_len = image_len1[0].tolist()
            # print('context_test:', context)
            # print('context2_test:', context2)

            # To device.
            texts = texts.to(GlobalConfig.device)
            image_features = image_features.to(GlobalConfig.device)
            text_lengths = text_lengths.to(GlobalConfig.device)

            # texts1 = list(target_utter_1)
            texts1 = texts.transpose_(0, 1)[-1]

            text_decoder.text_generation(context2, texts1, id2word, output_file, image_features, image_len, context_visual_know, text_visual_know)
        exit()

        command = "sh eval_2.sh " + output_file_test_name
        # ref_file + "<" + cand_file + "> " + temp
        try:
            t = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode()

        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)
    to_hidden.train()
    text_decoder.train()
    print('bleu_1_test:', t)
    output_file.close()

    return t
