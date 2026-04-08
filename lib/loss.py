import torch
import random
import numpy as np
from config import GlobalConfig, DatasetConfig
from constant import SOS_ID, EOS_ID
from model import ToHidden, TextDecoder
from model import Similarity
from util import mask_nll_loss, get_mask
import copy
import warnings
import subprocess
import sys


warnings.filterwarnings('ignore')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# smooth = SmoothingFunction()
smooth = SmoothingFunction() 

def text_loss(to_hidden: ToHidden, text_decoder: TextDecoder, id2word, text_length: int, context, target, target_length, hiddens, output_file, output_file_name, teacher_forcing_ratio, encode_knowledge_func=None):
    """Text loss.

    Args:
        to_hidden (ToHidden): Context to hidden.
        text_decoder (TextDecoder): Text decoder.
        text_length (int): Text length. #--20
        context: Context (batch_size, ContextEncoderConfig.output_size).
        target: Target (batch_size, dialog_text_max_len)
        target_length: Target length (batch_size, ).--#真实长度
        encode_knowledge_func (optional): Knowledge encoding function.
        hiddens: (seq_len, batch, num_directions * hidden_size).---(batch, 300)

    Returns:
        loss: Loss.
        n_totals: Number of words which produces loss.

    """
    batch_size = context.size(0)
    # print('batch_size:', np.shape(batch_size))
    loss = 0
    n_totals = 0
    # text_length:最长的length[20]；target_length:目标utterance实际的length
    mask2 = get_mask(text_length, target_length)
    # print('mask2: ', mask2)
    # print('mask2_shape: ', np.shape(mask2))    
    # mask2 = mask1.bool()
    # print('mask1:', np.shape(mask1))
    mask = mask2.transpose(0, 1)
    # print('mask3:', np.shape(mask))
    # (text_length, batch_size)
    # 使用context初始化decoder
    hidden = to_hidden(context).to(GlobalConfig.device)
    word = SOS_ID * torch.ones(batch_size, dtype=torch.long)
    word = word.to(GlobalConfig.device)
    # 返回全为1的64维向量---全为0
    # print('word:', word)
    # print('word:', np.shape(word))
    word1 = word
    # before--transpose: torch.Size([64, 30])
    # print('before--transpose:', np.shape(target))
    # after--transpose: torch.Size([30, 64])

    target = target.transpose(0, 1)
    # print('target:', target)
    # print('target:', np.shape(target))
    # use_teacher_forcing = random.random() < teacher_forcing_ratio
    use_teacher_forcing = teacher_forcing_ratio

    # print('use_teacher_forcing:', use_teacher_forcing)
    target1 = target
    # print('text_length2:', text_length)
    # print('target:', np.shape(target))
    # print('target1:', np.shape(target1))

    # print('mask2:', np.shape(mask))
    # ==注释掉下边部分
    # for i in range(text_length):
    #     # print('i----:', i)
    #     # print('i----:', np.shape(i))
    #     for each_ele_index in range(len(target[i])):
    #         each_ele = target[i][each_ele_index]
    #         if each_ele>=4892:
    #             target1[i][each_ele_index] = 3
    # ==注释掉上边部分
    # print('target1: ', np.shape(target1))

    all_tokens = torch.zeros((text_length, batch_size), dtype=torch.long)
    all_tokens = all_tokens.to(GlobalConfig.device)
    all_scores = torch.zeros((text_length, batch_size))
    all_scores = all_scores.to(GlobalConfig.device)


    # (text_length, batch_size)
    # 直接截至到20个单词
    # for i in range(text_length):
    for i in range(20):
        # word1
        # hidden:上一时刻隐藏层的状态[初始状态为context]
        # hiddens: context和know的级联
        # encode_knowledge_func为空
        output, hidden = text_decoder(word1, hidden, hiddens,
                                      encode_knowledge_func)
        # output, hidden = text_decoder(word, hidden, hiddens,
        #                               encode_knowledge_func)
        # output:  torch.Size([64, 4892])
        # hidden:  torch.Size([64, 512])        
        # print('output: ', np.shape(output))
        # print('hidden: ', np.shape(hidden))

        # for each_ele_index in range(len(target[i])):
        #     each_ele = target[i][each_ele_index]
        #     if each_ele>=4892:
        #         target1[i][each_ele_index] = 3
        # print('target1: ', np.shape(target1))
        # 返回的是两个值：一个是每一行最大值的tensor组，另一个是最大值所在的位置
        score, word = torch.max(output, dim=1)
        all_tokens[i], all_scores[i] = word, score
        # print('all_tokens[i]: ', all_tokens[i])
        # print('all_scores[i]: ', all_scores[i])
        # print('all_tokens[i]_shape: ', np.shape(all_tokens[i]))
        # print('all_scores[i]_shape: ', np.shape(all_scores[i]))

        # mask_loss, n_total = mask_nll_loss(output, target1[i], mask[i].bool())
        mask_loss, n_total = mask_nll_loss(output, target1[i], mask[i])
        # print('mask_loss:', np.shape(mask_loss))
        # print('n_total:', np.shape(n_total))
        # exit()
        # ===注释到line86
        if use_teacher_forcing:
            # 求tensor中某个dim的k个最大值
            # 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标
            topv, topi = output.topk(1)
            # print('topv: ', topv)
            # print('topi: ', topi)
            # word = topi.squeeze(1).detach()
            # 返回的 Variable 不会梯度更新---detach
            word1 = topi.squeeze(1).detach()
            # print('word1: ', word1)
            # print('-----valid------')
        else:
            word1 = target1[i]
            # print('-----train------')
        # train时，使用groundtruth中的单词作为train的输入
        # word1 = target1[i]

        # loss += mask_loss
        loss = loss + mask_loss
        n_totals = n_totals +  n_total
        # n_totals += n_total
    # print('all_tokens: ', all_tokens)
    # print('all_scores: ', all_scores)
    # print('all_tokens_shape: ', np.shape(all_tokens))
    # print('all_scores_shape: ', np.shape(all_scores))

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
    # print('str_pred_batch: ', str_pred_batch)        
    # print('str_pred_batch: ', str_pred_batch)        
    # command1 =  "python tools/split.py " + output_file_name
    # command2 =  "python tools/convert.py src text true pred " + output_file_name + ".true"
    # command3 =  "python tools/convert.py ref text true pred " + output_file_name + ".true"
    # command4 =  "python tools/convert.py tst text true pred " + output_file_name + ".pred"
    # command5 =  "perl tools/mteval-v14.pl -s " + output_file_name + ".text_src.xml" + "-r" + output_file_name + "text_ref.xml" + "-t" +'text_tst.xml'


    # NLTK计算方法
    # bleu_list = []---更改bleu计算方法
    # bleu1 = sentence_bleu(str_true_batch, str_pred_batch, weights=(1, 0, 0, 0))
    # bleu2 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0.5, 0.5, 0,0))
    # bleu3 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0.33, 0.33, 0.33, 0))
    # bleu4 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0.25, 0.25, 0.25, 0.25))    
    # # bleu1 = sentence_bleu(str_true_batch, str_pred_batch, weights=(1,0, 0,0), smoothing_function=smooth.method1)
    # # bleu2 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0,1, 0,0), smoothing_function=smooth.method1)
    # # bleu3 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0,0, 1,0), smoothing_function=smooth.method1)
    # # bleu4 = sentence_bleu(str_true_batch, str_pred_batch, weights=(0,0, 0,1), smoothing_function=smooth.method1)
    # bleu_list = [bleu1, bleu2, bleu3, bleu4]

    # return loss, n_totals, bleu_1
    return loss, n_totals



