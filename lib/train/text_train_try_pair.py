from datetime import datetime
from itertools import chain
from os.path import isfile
from typing import List, Dict
# import datetime
import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
import scipy.sparse as sp
from config import TextTrainConfig, GlobalConfig
from config.model_config import SimpleTextDecoderConfig
from constant import TEXT_TASK
from dataset import Dataset
from lib.test import text_test
from lib.valid import text_valid
from model import TextDecoder, ToHidden
import numpy as np
import random
import scipy.sparse as sp
from constant import PAD_ID, EOS_ID
import warnings

warnings.filterwarnings('ignore')
import copy
import subprocess
import sys

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
                each_target.append('</s>')

        if len(each_target) > 30:  # max_len-30
            each_target[:29].append('</s>')

        each_target_1 = " ".join(each_target)  # .replace(' .','.')

        batch_target.append(each_target_1)

    return batch_target

# origin
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


def text_train(
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        model_file: str,
        vocab: Dict[str, int],
        embed_init=None
):
    vocab_size = 4892  # num_node
    # 定义解码器的属性信息
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TextTrainConfig.batch_size,
        shuffle=False,
        num_workers=0)

    # Model.
    vocab_size = 4892  # num_node
    # 定义解码器的属性信息
    text_decoder_config = SimpleTextDecoderConfig(vocab_size, embed_init)
    # 讲输出映射到隐藏层
    to_hidden = ToHidden(text_decoder_config)
    to_hidden = to_hidden.to(GlobalConfig.device)
    text_decoder = TextDecoder(text_decoder_config)
    text_decoder = text_decoder.to(GlobalConfig.device)

    # Model parameters.
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        to_hidden,
        text_decoder
    ]]))

    optimizer = Adam(params, lr=TextTrainConfig.learning_rate)
    # optimizer = SGD(params, lr=TextTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None
    max_valid_bleu = None

    # Load saved state.
    if isfile(model_file):
        state = torch.load(model_file)
        to_hidden.load_state_dict(state['to_hidden'])
        text_decoder.load_state_dict(state['text_decoder'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_id = state['epoch_id']
        min_valid_loss = state['min_valid_loss']
        max_valid_bleu = state['max_valid_bleu']

    # Loss.
    sum_loss = 0
    bad_loss_cnt = 0
    # batch_size = 16 #32

    to_hidden.train()
    text_decoder.train()

    id2word: List[str] = [None] * 4892
    index = 0
    # idx和vocab中的id是一致的
    # vocab中的id和glove中的id是一致的[glove是按照vocab中排列的]
    for word in vocab.keys():
        wid = vocab[word]
        id2word[wid] = word
        index = index + 1
        if index == 4892:
            break

    output_file_train = open('text_177_train.out', 'w',encoding='UTF-8')

    finished = False
    # num_iterations = 10000
    for epoch_id in range(epoch_id, TextTrainConfig.num_iterations):
        for batch_id, train_data in enumerate(train_data_loader):
            # print('batch_id:', batch_id) 
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Set gradients to 0.--每个batch 参数初始化为0
            optimizer.zero_grad()  # # 将模型的参数梯度初始化为0
            # texts, image_features, text_lengths, batch_index, text_venues, image_len1, context, target_utter_1 = train_data
            texts, image_features, text_lengths, batch_index, text_venues, image_len1, visual_know_list = train_data
        #     continue
        # print('begin---valid----')
        # text_valid(
        #     vocab,
        #     to_hidden,
        #     text_decoder,
        #     valid_dataset,
        #     text_decoder_config.text_length)
        # print('begin---test----')
        # bleu_1_test = text_test(
        #     to_hidden,
        #     text_decoder,
        #     test_dataset,
        #     text_decoder_config.text_length,
        #     vocab)

    #  注释掉以下内容
    #         image_len = image_len1[0].tolist()
    #
    #         context = target_id2word(texts, id2word)
    #         context_visual_know = visual_know(visual_know_list)
    #         context2, text_visual_know = context_know(context, text_venues[0], context_visual_know)  # origin
    #
    #         # context2 = context_know(context, text_venues[0]) # origin
    #
    #         texts = texts.to(GlobalConfig.device)
    #         image_features = image_features.to(GlobalConfig.device)
    #         # 64,11
    #         # 是真实的文本长度
    #         text_lengths = text_lengths.to(GlobalConfig.device)
    #         texts1 = texts.transpose_(0, 1)[-1]
    #
    #         loss_gener = text_decoder(context2, texts1, id2word, output_file_train, image_features, image_len, context_visual_know, text_visual_know)
    #         sum_loss = loss_gener
    #         sum_loss.backward(retain_graph=True)
    #         optimizer.step()
    #         if (batch_id + 1) % TextTrainConfig.print_freq == 0: # 100
    #             cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #             print('epoch: {} \tbatch: {} \tloss: {} \tloss_gener: {}  \ttime: {}'.format(
    #                 epoch_id + 1, batch_id + 1, sum_loss, loss_gener, cur_time))
    #
    #         loss_dir = '/root/Model_data_dump12/loss_177.txt'
    #
    #         f_loss = open(loss_dir, 'a')
    #         f_loss.write(str(sum_loss) + '\r\n')
    #         f_loss.close()
    #         # continue
    #         # 从原来设置的1000---100
    #         if (batch_id + 1) % TextTrainConfig.valid_freq == 0:  # 400
    #         # if (batch_id + 1) % 1 == 0:  # 400
    #             # 一个epoch验证一次
    #             bleu_1_valid = text_valid(
    #                 vocab,
    #                 to_hidden,
    #                 text_decoder,
    #                 valid_dataset,
    #                 text_decoder_config.text_length)
    #
    #             cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #             print('time: {}'.format(cur_time))
    #             print('valid_bleu_1: %s ' % (str(bleu_1_valid)))
    #             bleu_1_valid = float(bleu_1_valid)
    #             # Save current best model./当模型最好时进行test
    #             if max_valid_bleu is None or bleu_1_valid > max_valid_bleu:
    #                 # min_valid_loss = valid_loss
    #                 max_valid_bleu = bleu_1_valid
    #                 save_dict = {
    #                     'task': TEXT_TASK,
    #                     'epoch_id': epoch_id,
    #                     'min_valid_loss': min_valid_loss,
    #                     'max_valid_bleu': max_valid_bleu,
    #                     'optimizer': optimizer.state_dict(),
    #                     'to_hidden':
    #                         to_hidden.state_dict(),
    #                     'text_decoder':
    #                         text_decoder.state_dict()
    #                 }
    #                 torch.save(save_dict, model_file)
    #                 print('Best model saved.')
    #
    #             # if bad_loss_cnt >= TextTrainConfig.patience: #5
    #
    #                 bleu_1_test = text_test(
    #                     to_hidden,
    #                     text_decoder,
    #                     test_dataset,
    #                     text_decoder_config.text_length,
    #                     vocab)
    #                 print('\ttime: {}'.format(cur_time))
    #                 print('test_bleu_1: %s' % (str(bleu_1_test)))
    #                 # exit()
    #             else:
    #                 print('bad_loss_cnt:', bad_loss_cnt)
    #                 bad_loss_cnt = bad_loss_cnt + 1
    #     # exit()
    # output_file_train.close()
