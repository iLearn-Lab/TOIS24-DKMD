"""Utilities."""

import pickle as pkl
from os.path import splitext, join
from typing import *
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')
from config import DatasetConfig, GlobalConfig
from constant import PAD_ID, EOS_ID


def save_pkl(obj, obj_name, pkl_file):
    """Save object to a .pkl file.

    Args:
        obj (object): Object.
        obj_name (str): Object name.
        pkl_file (str): Pickle file name.

    Returns:
        None

    """
    print('Saving {} to {}...'.format(obj_name, pkl_file))
    with open(pkl_file, 'wb') as file:
        pkl.dump(obj, file)
    print('Saved.')


def load_pkl(pkl_file):
    """Load object from a .pkl file.

    Args:
        pkl_file (str): Pickle file name.

    Returns:
        obj (object): Object.

    """
    print('Loading {} ...'.format(pkl_file))
    with open(pkl_file, 'rb') as file:
        obj = pkl.load(file)
    print('Loaded.')
    return obj

def load_pkl2(pkl_file):
    """Load object from a .pkl file.

    Args:
        pkl_file (str): Pickle file name.

    Returns:
        obj (object): Object.

    """
    print('Loading {} ...'.format(pkl_file))
    with open(pkl_file, 'rb') as file:
        obj = pkl.load(file)
    print('Loaded.')
    obj2 = list(obj.values())
    return obj2


def pad_or_clip_text(text: List[int], max_text_len, max_img_len):
                    #  max_len: int) -> Tuple[List[int], int]:
    """Pad or clip text.

    Args:
        text (List[int]): Word id list.
        max_len (int): Maximum length.

    Returns:
        Tuple[List[int], int]: Padded word list of length max_len and its
        actual length.

    """
    # 分离text和img--text padding/img padding
    utter_text = deepcopy(text)
    text_list = []
    img_list = []
    for each_text in utter_text:
        if each_text <4892:
            text_list.extend([each_text])
        else:
            img_list.extend([each_text])
    # 文本后边加一个EOS_ID
    if len(text_list) > max_text_len - 1:
        text_list = text_list[:max_text_len - 1]
    text_list.append(EOS_ID)
    text_len_real = len(text_list)
    text_list += [PAD_ID] * (max_text_len - len(text_list))

    if len(img_list) > max_img_len - 1:
        img_list = img_list[:max_img_len - 1]
    img_list.append(EOS_ID)
    img_list += [PAD_ID] * (max_img_len - len(img_list))

    text_list.extend(img_list)
    text_len = len(text_list)
    return text_list, text_len_real

# def pad_or_clip_text(text: List[int],
#                      max_len: int) -> Tuple[List[int], int]:
#     # # EOS_ID是在每条内容后边加上结束符，如果长度不是指定的长度，那么使用PAD_ID来补全
#     # # deepcopy一旦复制出来了，就是独立的了
#     text = deepcopy(text)
#     if len(text) > max_len - 1:
#         text = text[:max_len - 1]
#     text.append(EOS_ID)
#     text_len = len(text)
#     text += [PAD_ID] * (max_len - len(text))
#     return text, text_len


def pad_or_clip_images(images: List[int],
                       max_len: int) -> Tuple[List[int], int]:
    """Pad or clip images.

    Args:
        images (List[int]): Image id list.
        max_len (int): Maximum length.

    Returns:
        Tuple[List[int], int]: Padded image list of length max_len and its
        actual length.

    """
    images = deepcopy(images)
    if len(images) > max_len:
        images = images[:max_len]
    num_images = len(images)
    images += [0] * (max_len - len(images))
    return images, num_images


def get_embed_init(glove: List[List[float]], vocab_size: int):
    """Get initial embedding.

    Args:
        glove (List[List[float]]): GloVe.
        vocab_size (int): Vocabulary size.

    Returns:
        Initial embedding (vocab_size, embed_size).

    """
    # fc1 = nn.Linear(2048, 300)
    # embed = [None] * vocab_size
    embed_word = [None] * len(glove)
    # embed_img = [None] * len(image_fea)
    # 为什么使用vocab_size当作idx呢，感觉应该使用vocab中的word对应的index才对呀-----搞清楚了，因为glove中embedding的顺序是按照vocab中的顺序来排放的，也就是vocab中的第一个单词 它的embedding也是在glove的第一个
    # for idx in range(vocab_size):
    #     vec = glove[idx]
    #     if vec is None:
    #         vec = torch.zeros(300)
    #         # 如果不是padding符号3(补全符号)的话，那么就是这个单词存在但是没有词库的表征--则随机初始化
    #         if idx != PAD_ID:
    #             # 如果不是padding的单词(即是不在词库中的词)，则随机初始化范围是[-0.25,0.25]
    #             vec.uniform_(-0.25, 0.25)
    #     else:
    #         vec = torch.tensor(vec)
    #     embed[idx] = vec
    for idx in range(len(glove)):
        vec = glove[idx]
        if vec is None:
            vec = torch.zeros(300)
            # 如果不是padding符号3(补全符号)的话，那么就是这个单词存在但是没有词库的表征--则随机初始化
            if idx != PAD_ID:
                # 如果不是padding的单词(即是不在词库中的词)，则随机初始化范围是[-0.25,0.25]
                vec.uniform_(-0.25, 0.25)
        else:
            vec = torch.tensor(vec)
        embed_word[idx] = vec   

    # for idx in range(len(image_fea)):
    #     vec = image_fea[idx]
    #     if vec is None:
    #         vec = torch.zeros(300)
    #         # 如果不是padding符号3(补全符号)的话，那么就是这个单词存在但是没有词库的表征--则随机初始化
    #         # if idx != PAD_ID:
    #         #     # 如果不是padding的单词(即是不在词库中的词)，则随机初始化范围是[-0.25,0.25]
    #         #     vec.uniform_(-0.25, 0.25)
    #     else:
    #         vec = torch.tensor(vec)
    #     embed_img[idx] = vec.long()  
    # for idx2 in range(len(glove), vocab_size):
    #     vec = image_fea[idx2-len(glove)]
    #     embed[idx2] = fc1(torch.LongTensor(vec))
    # torch.stack(tensors,dim=0,out=None)→ Tensor
    # embed中对应词库中每个单词的表征
    embed_word = torch.stack(embed_word, dim=0)
    # embed_img = torch.stack(embed_img)

    # return embed_word, embed_img
    return embed_word



def get_product_path(image_name: str):
    """Get product path from a given image name.

    Args:
        image_name (str): Image name.

    Returns:
        str: Corresponding product file name.

    """
    product_path = join(DatasetConfig.product_data_directory,
                        splitext(image_name)[0] + '.json')
    return product_path


def nll_loss(inp, target):
    cross_entropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.mean()
    loss = loss.to(GlobalConfig.device)
    return loss


def mask_nll_loss(inp, target, mask):
    # print('inp = {}'.format(inp))
    # print('target = {}'.format(target))
    # print('mask = {}'.format(mask))
    # print('inp:', inp)
    # print('target:', target)
    # print('mask:', mask)
    n_total = mask.sum()
    # print('n_total:', n_total)
    # print('n_total:', np.shape(n_total))

    if n_total.item() == 0:
        return 0, 0
    # 64, 4892----[1,64]
    CE1 = torch.gather(inp, 1, target.view(-1, 1)).squeeze(1)
    # print('CE1:', CE1)
    # print('CE1:', np.shape(CE1))
    cross_entropy = -torch.log(CE1)
    # print('cross_entropy:', cross_entropy)
    # print('cross_entropy:', np.shape(cross_entropy))
    # 根据mask 只保留真实存在的loss
    # mask.byte()---将mask映射成二元值
    loss = cross_entropy.masked_select(mask.byte()).mean()
    # print('cross_entropy:', np.shape(cross_entropy))
    # print('mask.byte(): ', mask.byte())
    # print('loss:', loss)
    # print('loss:', np.shape(loss))
    loss = loss.to(GlobalConfig.device)
    n_total1 = n_total.item()
    return loss, n_total1


def get_mask(length: int, target_length):
    """Get mask.

    Args:
        length (int): Length.
        target_length: Target length (batch_size, ).

    Returns:
        mask: Mask (batch_size, length).

    """
    # torch.arange(start=1.0,end=6.0)的结果不包括end
    # 返回tensor的一个新视图，单个维度扩大为更大的尺寸
    # torch.arange(start=1.0,end=6.0)的结果不包括end 
    mask = torch.arange(length, device=GlobalConfig.device).expand(
        target_length.size(0), length) < target_length.unsqueeze(1)
    # True/False
    # print('mask1: ', mask)
    # 64, 20
    # print('mask1_shape: ', np.shape(mask))

    # return mask.bool()
    return mask



def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """

    Source: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.softmax(vector)`` does not work if some elements
    of ``vector`` should be masked.  This performs a softmax on just the
    non-masked portions of ``vector``.  Passing ``None`` in for the mask is
    also acceptable; you'll just get a regular softmax. ``vector`` can have
    an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions
    than ``vector``, we will unsqueeze on dimension 1 until they match.  If
    you need a different unsqueezing of your mask, do it yourself before
    passing the mask into this function. If ``memory_efficient`` is set to
    true, we will simply use a very large negative number for those masked
    positions so that the probabilities of those positions would be
    approximately 0. This is not accurate in math, but works for most cases
    and consumes less memory. In the case that the input vector is completely
    masked and ``memory_efficient`` is false, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last
    layer of a model that uses categorical cross-entropy loss. Instead,
    if ``memory_efficient`` is true, this function will treat every element
    as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the
            # mask, we zero these out.
            result = torch.nn.functional.softmax(torch.mul(vector, mask),
                                                 dim=dim)
            result = torch.mul(result, mask)
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def to_str(o):
    if isinstance(o, dict):
        res = []
        for key, val in o.items():
            res.append(key)
            res.append(to_str(val))
        return ' '.join(res)
    elif isinstance(o, list):
        return ' '.join([to_str(x) for x in o])
    elif isinstance(o, str):
        return o
    return str(o)
