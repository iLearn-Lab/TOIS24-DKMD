"""Data models.

Data models:
    * Utterance
    * Product
    * TidyUtterance

"""

from tokenize import String
from typing import List, Dict, Any

from config import DatasetConfig
from util import pad_or_clip_text, pad_or_clip_images


class Utterance():
    """Utterance data model.

    Attributes:
        speaker (int): Speaker.
                       0 (USER_SPEAKER) for user, 1 (SYS_SPEAKER) for system.
        utter_type (int): Utterance type.(这个有没有具体的定义呢，具体分哪些类型呢)
        text (str): Text.
        pos_images (List[int]): Positive images.
        neg_images (List[int]): Negative images.

    """
    def __init__(self, speaker: int, utter: String, utter_type: int, text: List[int], text_vec:list, phrase:list, domain:list, venue:int,
                pos_images: List[int], origin_imgs: List[int]):
        self.speaker: int = speaker
        self.utter = utter
        self.utter_type: int = utter_type
        self.text: List[int] = text
        self.text_vec = text_vec
        self.phrase = phrase
        self.domain = domain
        self.venue = venue
        self.pos_images: List[int] = pos_images
        self.origin_imgs: List[int] = origin_imgs

    def __repr__(self):
        return str((self.speaker, self.utter, self.utter_type, self.text, self.text_vec, self.phrase, self.domain, self.venue,
                    self.pos_images, self.origin_imgs))
        # return str((self.speaker, self.utter_type, self.text, self.text_vec, self.domain,
        #             self.pos_images, self.neg_images))


class Product():
    """Product data model.

    Attributes:
        product_name (str): Product name, which is the name of the .json file.
        attribute_dict (Dict[str, Any]): Attribute dictionary.

    """

    def __init__(self, product_name: str, attribute_dict: Dict[str, Any]):
        self.product_name = product_name
        self.attribute_dict = attribute_dict


class TidyUtterance():
    """Tidy utterance data model.

    Attributes:
        utter_type (int): Utterance type.
        text (List[int]): Text.
        text_len (int): Text length.
        pos_images (List[int]): Positive images.
        pos_images_num (int): Number of positive images.
        neg_images (List[int]): Negative images.
        neg_images_num (int): Number of negative images.

    """

    def __init__(self, utter: Utterance):
        self.utter_type: int = utter.utter_type
        self.utter = utter.utter  # do not padding or clip
        self.text_vec = utter.text_vec
        self.phrase = utter.phrase
        self.domain = utter.domain
        self.venue = utter.venue
        self.origin_imgs = utter.origin_imgs
        # 一句话中最多有30个单词
        self.text, self.text_len = pad_or_clip_text(
            utter.text, DatasetConfig.max_text_len, DatasetConfig.max_img_len)
        # print('text_len:', self.text_len)
        # self.text, self.text_len = pad_or_clip_text(
        #     utter.text, DatasetConfig.dialog_text_max_len)
        self.pos_images, self.pos_images_num = pad_or_clip_images(
            utter.pos_images, DatasetConfig.pos_images_max_num)
        # self.neg_images, self.neg_images_num = pad_or_clip_images(
            # utter.neg_images, DatasetConfig.neg_images_max_num)

    # def __repr__(self):
    #     return str((self.utter_type,
    #                 self.text, self.text_vec, self.domain, self.venue, self.text_len,
    #                 self.pos_images, self.pos_images_num,
    #                 self.neg_images, self.neg_images_num))
    def __repr__(self):
        return str((self.utter_type, self.utter,
                    self.text, self.text_vec, self.phrase, self.domain, self.venue, self.text_len,
                    self.pos_images, self.pos_images_num,
                    self.origin_imgs))