from typing import List

import torch
import numpy as np
from config import GlobalConfig, DatasetConfig
from constant import SOS_ID, EOS_ID
from model import Similarity, ToHidden, TextDecoder
from util import get_mask
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import warnings
warnings.filterwarnings('ignore')
smooth = SmoothingFunction()

def recommend_eval(similarity: Similarity, batch_size: int, context,
                   pos_products, neg_products):
    """Recommend Evaluation.

    Args:
        similarity (Similarity):
        batch_size (int):
        context: Context.
        pos_products: Positive products. (num_pos_products, pos_images,
                      pos_product_texts, pos_product_text_lengths)
        neg_products: Negative products. (num_neg_products, neg_images,
                      neg_product_texts, neg_product_text_lengths)

    """

    (num_pos_products, pos_images, pos_product_texts,
     pos_product_text_lengths) = pos_products
    (num_neg_products, neg_images, neg_product_texts,
     neg_product_text_lengths) = neg_products

    # Sizes:
    # num_pos_products: (batch_size, )
    # pos_images: (batch_size, pos_images_max_num, 3, image_size, image_size)
    # pos_product_texts: (batch_size, pos_images_max_num, product_text_max_len)
    # pos_product_text_lengths: (batch_size, pos_images_max_num)
    #
    # num_neg_products: (batch_size, )
    # neg_images: (batch_size, neg_images_max_num, 3, image_size, image_size)
    # neg_product_texts: (batch_size, neg_images_max_num, product_text_max_len)
    # neg_product_text_lengths: (batch_size, neg_images_max_num)

    # num_pos_products = num_pos_products.to(GlobalConfig.device)
    pos_images = pos_images.to(GlobalConfig.device)
    pos_product_texts = pos_product_texts.to(GlobalConfig.device)
    pos_product_text_lengths = pos_product_text_lengths.to(GlobalConfig.device)
    pos_images.transpose_(0, 1)
    pos_product_texts.transpose_(0, 1)
    pos_product_text_lengths.transpose_(0, 1)
    # pos_images: (pos_images_max_num, batch_size, 3, image_size, image_size)
    # pos_product_texts: (pos_images_max_num, batch_size, product_text_max_len)
    # pos_product_text_lengths: (pos_images_max_num, batch_size)

    num_neg_products = num_neg_products.to(GlobalConfig.device)
    neg_images = neg_images.to(GlobalConfig.device)
    neg_product_texts = neg_product_texts.to(GlobalConfig.device)
    neg_product_text_lengths = neg_product_text_lengths.to(GlobalConfig.device)
    neg_images.transpose_(0, 1)
    neg_product_texts.transpose_(0, 1)
    neg_product_text_lengths.transpose_(0, 1)
    # neg_images: (neg_images_max_num, batch_size, 3, image_size, image_size)
    # neg_product_texts: (neg_images_max_num, batch_size, product_text_max_len)
    # neg_product_text_lengths: (neg_images_max_num, batch_size)

    pos_cos_sim = similarity(context, pos_product_texts[0],
                             pos_product_text_lengths[0],
                             pos_images[0])
    # Mask.
    mask = get_mask(DatasetConfig.neg_images_max_num, num_neg_products)
    mask = mask.transpose(0, 1).long()
    # (neg_images_max_num, batch_size)

    rank = torch.zeros(batch_size, dtype=torch.long).to(GlobalConfig.device)
    for i in range(DatasetConfig.neg_images_max_num):
        neg_cos_sim = similarity(context, neg_product_texts[i],
                                 neg_product_text_lengths[i],
                                 neg_images[i])
        rank += torch.lt(pos_cos_sim, neg_cos_sim).long() * mask[i]

    num_rank = [0] * (DatasetConfig.neg_images_max_num + 1)
    for i in range(batch_size):
        num_rank[rank[i]] += 1

    return torch.tensor(num_rank).to(GlobalConfig.device)


def text_eval(to_hidden: ToHidden, text_decoder: TextDecoder, text_length: int,
              id2word: List[str], context, target1,  hiddens, output_file, output_file_name,
              encode_knowledge_func=None):
    """Text loss.

    Args:
        to_hidden (ToHidden): Context to hidden.
        text_decoder (TextDecoder): Text decoder.
        text_length (int): Text length.
        id2word (List[str]): Word id to str.
        context: Context (batch_size, ContextEncoderConfig.output_size).
        target: Target (batch_size, dialog_text_max_len)
        encode_knowledge_func (optional): Knowledge encoding function.
        output_file (optional): Output file.

    """
    batch_size = context.size(0)

    hidden = to_hidden(context)
    word = SOS_ID * torch.ones(batch_size, dtype=torch.long)
    word = word.to(GlobalConfig.device)
    # print('word:', word)    
    # (text_length, batch_size)
    # 传进来之前取的-1====>是取了每个context的最后一个
    target = target1.transpose(0, 1)
    # print('test_target:', target)
    # print('test_target_shape:', np.shape(target))

    # target2 = target
    # for i in range(text_length):
    #     for each_ele_index in range(len(target[i])):
    #         each_ele = target[i][each_ele_index]
    #         if each_ele>=4892:
    #             target[i][each_ele_index] = 3
    # batch_size = target.size(1)    
    # print('batch_size:', batch_size)


    # print('text_length:', text_length)
    # print('target:', np.shape(target))
    # (text_length, batch_size)

    all_tokens = torch.zeros((text_length, batch_size), dtype=torch.long)
    all_tokens = all_tokens.to(GlobalConfig.device)
    all_scores = torch.zeros((text_length, batch_size))
    all_scores = all_scores.to(GlobalConfig.device)

    # for i in range(text_length):
    for i in range(20):
        # 单词使用预测
        output, hidden = text_decoder(word, hidden,
                                      hiddens.float(), encode_knowledge_func)
        score, word = torch.max(output, dim=1)
        all_tokens[i], all_scores[i] = word, score

    str_pred_batch = []
    str_true_batch = []

    for j in range(batch_size):
        # print('j:', j)
        str_pred = []
        str_true = []
        for i in range(text_length):
            if all_tokens[i][j] == EOS_ID:
                break
            word = id2word[all_tokens[i][j]]
            str_pred.append(word)

        for i in range(text_length):
            # print('i:', i)
            if target[i][j] == EOS_ID:
                break
            # print('target[i][j]:', target[i][j])
            word = id2word[target[i][j]]
            str_true.append(word)
        line = "{}\t{}".format(' '.join(str_pred), ' '.join(str_true))
        if output_file:
            output_file.write(line + '\n')
        else:
            print(line)
        str_pred_batch.extend(str_pred)
        str_true_batch.extend(str_true)
    
    # command =  "sh eval_2.sh " + output_file_name
    # try:
    #     # subprocess.call(command, shell=True) stdout=subprocess.PIPE).stdout
    #     # output = subprocess.run(command, shell=True).returncode
    #     bleu_1 = float((subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout).decode("gbk"))
    #     # output = subprocess.run(command, shell=True).args
    #     print(bleu_1)
    #     print(type(bleu_1))

    # except OSError as e:
    #     print("Execution failed:", e, file=sys.stderr)
    # bleu_list = []
    # bleu1 = sentence_bleu(str_true_batch, str_pred_batch, weights=(1, 0, 0, 0))
    # bleu2 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0.5, 0.5, 0,0))
    # bleu3 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0.33, 0.33, 0.33, 0))
    # bleu4 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0.25, 0.25, 0.25, 0.25))      
    # bleu1 = sentence_bleu(str_true_batch, str_pred_batch, weights=(1,0, 0,0), smoothing_function=smooth.method1)
    # bleu2 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0,1, 0,0), smoothing_function=smooth.method1)
    # bleu3 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0,0, 1,0), smoothing_function=smooth.method1)
    # bleu4 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0,0, 0,1), smoothing_function=smooth.method1)
    # bleu_list = [bleu1, bleu2, bleu3, bleu4]
    
    # print('bleu_1: %s, bleu_2: %s, bleu_3: %s, bleu_4: %s,' % (str(bleu_list[0]), str(bleu_list[1]), str(bleu_list[2]), str(bleu_list[3])))

    return all_tokens
    # return bleu_list

