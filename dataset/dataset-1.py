"""Dataset module."""
import json
from os.path import join, isfile
from typing import List, Dict

import torch
from torch.utils import data

from config import DatasetConfig
from constant import UNK_ID, PAD_ID, EOS_ID
from util import pad_or_clip_text, get_product_path
import numpy as np
import clip
from PIL import Image
from config import TextTrainConfig, GlobalConfig


class Dataset(data.Dataset):
    """Dataset class."""

    # Constants.
    EMPTY_IMAGE = torch.zeros(3, DatasetConfig.image_size,
                              DatasetConfig.image_size)
    EMPTY_PRODUCT_TEXT = [EOS_ID] + [PAD_ID] * (
            DatasetConfig.product_text_max_len - 1)

    def __init__(self, task: int, context_dir, 
                 dialog_vocab: Dict[str, int],
                 train_context_index
                 ):
        self.task: int = task
        self.context_dir = context_dir
        self.dialog_vocab: Dict[str, int] = dialog_vocab
        self.train_context_index = train_context_index
        # index_know_text
        path_index_know = '/root/autodl-tmp/dataset/index_know.npy'
        self.index_know_dict = np.load(path_index_know, allow_pickle=True).tolist()
        # name_know_text
        path_name_know = '/root/autodl-tmp/dataset/name_know.npy'
        self.name_know_dict = np.load(path_name_know, allow_pickle=True).tolist()
        # img-name_know-name
        path_img_db = '/root/autodl-tmp/dataset/img_db_name.npy'
        self.image_content = np.load(path_img_db, allow_pickle=True).tolist()
        self.path_img_fea = '/root/autodl-tmp/Img_Feature/'
        self.path_img_jpg = '/root/autodl-tmp/Image/'
        self.img_model, self.img_preprocess = clip.load("ViT-B/32", device=GlobalConfig.device)
        # self.img_model = self.img_model.to(GlobalConfig.device)
        # self.img_preprocess = self.img_preprocess.to(GlobalConfig.device)

    def __len__(self):
        # return len(self.dialogs)
        return len(self.train_context_index)

    def __getitem__(self, index: int):
        """ Get item for a given index.

        Args:
            index (int): item index.

        Returns:
            - INTENTION_TASK, TEXT_TASK, KNOWLEDGE_STYLETIP_SUBTASK and
              KNOWLEDGE_CELEBRITY_SUBTASK
                texts: Texts (dialog_context_size + 1, dialog_text_max_len).
                text_lengths: Text lengths (dialog_context_size + 1, ).
                images: Images (dialog_context_size + 1, pos_images_max_num, 3,
                               image_size, image_size).
                utter_type (int): The type of the last user utterance.

            - RECOMMEND_TASK
                context_dialog:
                    texts: Texts (dialog_context_size + 1, dialog_text_max_len).
                    text_lengths: Text lengths (dialog_context_size + 1, ).
                    images: Images (dialog_context_size + 1, pos_images_max_num,
                                    3, image_size, image_size).
                    utter_type (int): The type of the last user utterance.
                pos_products:
                    num_pos_products (int): Number of positive products.
                    pos_images: Positive images
                                (pos_images_max_num, 3, image_size, image_size).
                    pos_product_texts: Positive product texts
                                     (pos_images_max_num, product_text_max_len).
                    pos_product_text_lengths: Positive product text lengths
                                          (pos_images_max_num, ).
                neg_products:
                    num_neg_products (int): Number of negative products.
                    neg_images: Negative images
                                (neg_images_max_num, 3, image_size, image_size).
                    neg_product_texts: Negative product texts
                                     (neg_images_max_num, product_text_max_len).
                    neg_product_text_lengths: Negative product text lengths
                                          (neg_images_max_num, ).

        """

        num_count = 0
        batch_index_list = []
        each_index = self.train_context_index[index]
        batch_index_list.append(each_index)
        # print('each_index:', each_index)
        path_dialog = self.context_dir + str(each_index) + '.npy'
        dialog_content = np.load(path_dialog,allow_pickle=True).tolist()

        # texts, texts_vec, text_lengths, utter_type, know_venue, img_num, text_utter, target_utter_1 = self._get_context_dialog(dialog_content)
        texts, texts_vec, text_lengths, utter_type, know_venue, img_num, visual_know_list = self._get_context_dialog(dialog_content)

        texts_batch = texts
        texts_vec_batch = texts_vec
        text_lengths_batch = text_lengths
        num_count = num_count + 1

        return texts_batch, texts_vec_batch, text_lengths_batch, batch_index_list, know_venue, img_num, visual_know_list

    def _get_context_dialog(self, dialog):
        """Get context dialog.

        Note: The last utterance of the context dialog is system response.

        Args:
            dialog (TidyDialog): Dialog.

        Returns:
            texts: Texts (dialog_context_size + 1, dialog_text_max_len).
            text_lengths: Text lengths (dialog_context_size + 1, ).
            images: Images (dialog_context_size + 1, pos_images_max_num, 3,
                           image_size, image_size).
            utter_type (int): The type of the last user utterance.

        """
        # print('dialog:', dialog)
        # Text.
        text_list: List[List[int]] = [utter.text for utter in dialog]
        text_length_list: List[int] = [utter.text_len for utter in dialog]

        # # context_raw_data
        # context_utter = []
        # target_utter = []
        # for utter_index in range(len(dialog)):
        #     each_utter = dialog[utter_index].utter
        #     # context
        #     if utter_index != (len(dialog)-1):
        #         if len(each_utter)>0:
        #             context_utter.append(each_utter)
        #             context_utter.append('<SEP>')
        #     else:
        #         target_utter.append(each_utter)
        # context_utter_1= " ".join(context_utter).replace("'","") #remove '
        # target_utter_1 = " ".join(target_utter).replace("'","")
        
        # each_len = len(context_utter_1.split(' '))
        # if each_len>29: # clip
        #     context_utter_1.split(' ')[:29].append('<SEP>')

        # each_target_len = len(target_utter_1.split(' '))
        # if each_target_len>20:
        #     target_utter_1.split(' ')[:20]

        text_venue_list = []
        for utter in dialog:
            utter_venue = utter.venue
            if len(utter_venue) > 0:
                for each_venue in utter_venue:
                    if each_venue not in text_venue_list:
                        text_venue_list.append(each_venue)
        # print('text_venue_list: ', text_venue_list)
        # text-img incor
        know_list = []
        for each_venue_index in text_venue_list:
            if each_venue_index in self.index_know_dict.keys():
                venue_know = self.index_know_dict[each_venue_index]
                know_list.extend([venue_know])  # 若包含多个venue，则一起放进去
        know_list = [" ".join(know_list)]
        if len(know_list) == 0:
            know_list = ['<pad>']

        # img_list
        imgs_list = []
        for utter_index in range(len(dialog)-1):  # context
            utter = dialog[utter_index]
            utter_img = utter.origin_imgs
            if len(utter_img) > 0:
                for each_img_index in range(len(utter_img)):
                    each_img = utter_img[each_img_index]
                    each_img2 = each_img.split('.')[0]
                    # imgs_list[each_img_index] = each_img2
                    imgs_list.extend([each_img2])
        if len(imgs_list)>2:
            imgs_list = imgs_list[:2] # 保留两张
            img_len = [2]
        else:
            img_len = [len(imgs_list)]
            padding_content = [0] * (2-len(imgs_list))
            imgs_list.extend(padding_content)


        # count image数目
        # f_img_num = open('img_num.txt', 'a')
        # f_img_num.write(str(len(imgs_list))+'\n')
        # f_img_num.close()

        # visual-img incor
        visual_know_list = []
        # imgs_list2 = ['7465b7b1084e295abe38b266f6e6ab99', '296a41a2744d82b81bf683a824923154']
        for each_img in imgs_list:  # self.image_content self.name_know_dict
            if each_img != 0:
                knowledge_key = self.image_content[each_img]
                if knowledge_key in self.name_know_dict.keys():
                    knowledge_value2 = self.name_know_dict[knowledge_key]
                    visual_know_list.extend([knowledge_value2])
            else:
                visual_know_list.append('<pad>')

        # text_vec_list = np.zeros([2, 2048])
        text_vec_list = torch.zeros(2, 512)
        for each_img_index in range(len(imgs_list)):
            each_img = imgs_list[each_img_index]
            if each_img!=0:
                # origin_img
                path_each_img = self.path_img_jpg + each_img + '.jpg'
                # path_each_img = '/root/autodl-tmp/example/example.png'
                each_img_1 = Image.open(path_each_img)
                each_img_pre = self.img_preprocess(each_img_1).unsqueeze(0).to(GlobalConfig.device)
                each_img_vec = self.img_model.encode_image(each_img_pre)  # 1,512
                # path_img_fea = self.path_img_fea + each_img.split('.')[0] + '.npy'
                # each_img_vec = np.load(path_img_fea, allow_pickle=True).tolist()
                text_vec_list[each_img_index] = each_img_vec  # 按照出现的顺序放，若是第二条utterance中有img，第一条没有，则第二条的放在开始

        # for utter_index in range(len(dialog) - 1):
        #     utter = dialog[utter_index]
        #     if utter.pos_images_num == 1:  # image exists
        #         each_img_vec = np.squeeze(utter.text_vec).tolist()[768:]
        #         text_vec_list[img_num] = each_img_vec  # 按照出现的顺序放，若是第二条utterance中有img，第一条没有，则第二条的放在开始
        #         img_num = img_num + 1


        # text_vec_list = np.zeros([2, 2048])
        # img_num = 0
        # for utter_index in range(len(dialog) - 1):
        #     utter = dialog[utter_index]
        #     if utter.pos_images_num == 1:  # image exists
        #         each_img_vec = np.squeeze(utter.text_vec).tolist()[768:]
        #         text_vec_list[img_num] = each_img_vec  # 按照出现的顺序放，若是第二条utterance中有img，第一条没有，则第二条的放在开始
        #         img_num = img_num + 1
        #
        # img_len = [img_num]

        texts = torch.stack(tuple([torch.tensor(text) for text in text_list]))

        # texts_vec = torch.stack([torch.LongTensor(text_vec) for text_vec in text_vec_list])
        texts_vec = text_vec_list

        text_lengths = torch.tensor(text_length_list)

        know_venue = know_list

        utter_type = dialog[-2].utter_type
        # return texts, texts_vec, text_lengths, utter_type, know_venue, img_len, context_utter_1, target_utter_1
        return texts, texts_vec, text_lengths, utter_type, know_venue, img_len, visual_know_list



    @staticmethod
    def _get_product_text(product_path):
        product_dict = json.load(open(product_path))
        texts = []
        for key, value in product_dict.items():
            # Note: Only a space is also empty.
            if value is not None and value != '' and value != ' ':
                texts.extend([key, value])
        return ' '.join(texts).lower()

    def get_attributes(self, product_id):
        keys = []
        values = []
        if product_id != 0:
            image_name = self.image_paths[product_id]
            product_path = get_product_path(image_name)
            if isfile(product_path):
                product_dict = json.load(open(product_path))
                for key, value in product_dict.items():
                    # Note: Only a space is also empty.
                    if value is not None and value != '' and value != ' ':
                        key = key.lower()
                        value = value.lower()
                        if key not in self.key_vocab:
                            continue
                        key_id = self.key_vocab[key]
                        if (key_id, value) in self.value_vocab:
                            value_id = self.value_vocab[(key_id, value)]
                            keys.append(key_id)
                            values.append(value_id)
        length = len(keys)
        pad = [0] * (len(self.key_vocab) - length)
        keys.extend(pad)
        values.extend(pad)
        return torch.tensor(keys), torch.tensor(values), torch.tensor(length)
