"""Raw data module."""
import copy
import json
from collections import Counter, namedtuple
from os import listdir
from os.path import isfile, isdir, join
from typing import List, Tuple, Dict, Optional
import os
import numpy as np
from nltk import word_tokenize
import copy
from config import DatasetConfig
from constant import DIALOG_PROC_PRINT_FREQ
from constant import SYS_SPEAKER, USER_SPEAKER
from constant import TRAIN_MODE, VALID_MODE, TEST_MODE
from constant import UNK_ID, SPECIAL_TOKENS, UNK_Img_ID
from dataset.model import Utterance
from util import load_pkl, save_pkl

# Custom types.
# 定义一个namedtuple类型CommonData，其中包含dialog_vocab, glove, image_url_id和image_paths属性，之后可以直接使用CommonData创建对象
# CommonData = namedtuple('CommonData',
#                         ['dialog_vocab', 'glove',
#                          'image_url_id', 'image_paths'])
# CommonData = namedtuple('CommonData',
#                         ['dialog_vocab', 'glove',
#                          'image_url_id', 'image_fea'])
CommonData = namedtuple('CommonData',
                        ['dialog_vocab', 'glove', 'image_fea'])
Dialog = List[Utterance]
predix = '/root/autodl-tmp/'
path_venue_index = predix+'/root/autodl-tmp/venue_index_dict2.json'
with open(path_venue_index) as json_file:
    venue_index_dict = json.load(json_file)

path_venue_domain = predix+'/root/autodl-tmp/venue_domain.json'
f_venue_domain = open(path_venue_domain, 'r')
venue_domain_dict = json.load(f_venue_domain)
venue_domain_list = list(venue_domain_dict.keys())

# phrase_index
path_phrase_dict_filter = '/root/UMD-master/data/phrase_add/phrase_dict_filter.npy'
phrase_dict_filter = np.load(path_phrase_dict_filter, allow_pickle=True).tolist()


# print(venue_index)
# print(venue_index.keys())
# 1845
# print(len(venue_index))
# exit()
err_list = []
def extract_venue(agent_act):
    venuename_index =10000
    venue_list = []
    venue_list_name = []
    if agent_act != {}:
        all_act_list = list(agent_act.keys())
        all_act_value = list(agent_act.values())
        for i in range(len(all_act_list)):
            act_specific = all_act_list[i]
            act_specific_value = all_act_value[i]
            # key分割
            key_venue = act_specific.split(':')
            if len(key_venue)>1:
                venuename_key = key_venue[0]
                venuename_value = key_venue[1].strip()
                if venuename_key=='venuename':
                    if venuename_value in venue_index_dict.keys(): 
                        venuename_index = venue_index_dict[venuename_value]
                        venue_list.extend([venuename_index])
                        venue_list_name.extend([venuename_value])
                    else:
                        err_list.append(venuename_value)
                        # print('----can not find venue index!------')
                        # print('venuename_value:', venuename_value)
                        # exit()
    else:
        venuename_index = venue_index_dict["unknown"]
        venue_list.extend([venuename_index])

    # 不存在这个键值
    if venuename_index == 10000:
        venuename_index = venue_index_dict["unknown"]
        venue_list.extend([venuename_index])

    return venue_list, venue_list_name

err_list_value = []
def extract_domain(venue_name):
    domain_dict = {'food':0, 'hotel':1, 'nightlife':2, 'mall':3, 'sightseeing':4}
    domain_list = []
    if len(venue_name) == 0:
        domain_list.extend([-1])
    else:
        for each_venue in venue_name:
            if each_venue in venue_domain_dict.keys():
                domain_index = domain_dict[venue_domain_dict[each_venue]]
                domain_list.extend([domain_index])
            else:
                if each_venue not in err_list_value:
                    err_list_value.extend([each_venue])
                domain_list.extend([-1])


    return domain_list

# 只保留出现两次之上的
def extract_phrase(phrase_list):
    phrase_list_filter = []
    if len(phrase_list) != 0:
        for each_phrase in phrase_list:
            if each_phrase in phrase_dict_filter.keys():
                phrase_list_filter.extend([phrase_dict_filter[each_phrase]])
    return phrase_list_filter


class RawData():
    """Raw data class.

    This class extracts raw data from some data files or directories or loads
    extracted .pkl files. Data is divided into two parts: common data and mode
    specific data. Common data is always loaded whenever the project runs. Mode
    specific data is loaded when the extracted item data file of specific
    mode doesn't exist.

    Common data:
        * dialog_vocab
        * glove
        * image_url_id
        * image_paths

    Mode specific data:
        * {train, valid, test}_dialogs

    Attributes:
        mode (int): Mode (TRAIN_MODE, VALID_MODE, TEST_MODE).
        dialog_vocab (Dict[str, int]): Dialog vocabulary, word -> index.-----?单词是不是要统计呢
        glove (List[Optional[List[float]]]): GloVe, index -> float vector. -----？index的统一(Vocab和glove)
        image_url_id (Dict[str, int]): Image url to index of `image_paths`.
                                       0 for unknown url.
        image_paths (List[str]): Image paths. image_paths[0] is None.

        train_dialogs (List[Dialog]): Train dialogs, if is train mode.
        valid_dialogs (List[Dialog]): Valid dialogs, if is valid mode.
        test_dialogs (List[Dialog]): Test dialogs, if is test mode.

    """

    def __init__(self, mode: int):
        # Note: For convenience, RawData loads common data (if exists) only if
        # mode is NONE_MODE

        self.mode: int = mode

        self.dialog_vocab: Dict[str, int] = None
        self.glove: List[Optional[List[float]]] = None  # ---类型的定义
        self.dimage_url_i: Dict[str, int] = None
        self.image_paths: List[str] = None

        if self.mode & TRAIN_MODE:
            self.train_dialogs: List[Dialog] = None
        if self.mode & VALID_MODE:
            self.valid_dialogs: List[Dialog] = None
        if self.mode & TEST_MODE:
            self.test_dialogs: List[Dialog] = None

        # Check if consistency of data files.
        # 若没报错，正常返回，则说明目前数据没事，要么都不存在，或者存在需要存在的部分
        RawData.check_consistency(mode)

        # Read existed extracted data files.--如有数据的话，直接load
        self.read_extracted_data()

        # If common data doesn't exist, then we need to get it.
        if not isfile(DatasetConfig.common_raw_data_file):
            common_data = RawData._get_common_data()
            self.dialog_vocab: Dict[str, int] = common_data.dialog_vocab
            self.glove: List[Optional[List[float]]] = common_data.glove
            self.image_fea: Dict[str, int] = common_data.image_fea

            # Save common data to a .pkl file.
            save_pkl(common_data, 'common_data',
                     DatasetConfig.common_raw_data_file)
            print('common_data_save!')
            # print(self.dialog_vocab)
            # exit()
            # 经过以上步骤，获取到common 数据--词库，golve词向量，图片的index-url，图片的index-名字
        # If mode specific data doesn't exist, then we need to get it.
        if self.mode & TRAIN_MODE:
            has_data_pkl = isfile(DatasetConfig.train_raw_data_file)

            if not has_data_pkl:
                self.train_dialogs = RawData._get_dialogs(TRAIN_MODE,
                                                          self.dialog_vocab,
                                                          self.image_fea)                
                # self.train_dialogs = RawData._get_dialogs(TRAIN_MODE,
                #                                           self.dialog_vocab,
                #                                           self.image_url_id)
                # exit()
                # Save common data to a .pkl file.
                print('Save traindata!')
                save_pkl(self.train_dialogs, 'train_dialogs',
                         DatasetConfig.train_raw_data_file)

        if self.mode & VALID_MODE:
            has_data_pkl = isfile(DatasetConfig.valid_raw_data_file)

            if not has_data_pkl:
                self.valid_dialogs = RawData._get_dialogs(VALID_MODE,
                                                          self.dialog_vocab,
                                                          self.image_fea)
                # Save common data to a .pkl file.
                print('Save validdata!')
                save_pkl(self.valid_dialogs, 'valid_dialogs',
                         DatasetConfig.valid_raw_data_file)

        if self.mode & TEST_MODE:
            has_data_pkl = isfile(DatasetConfig.test_raw_data_file)

            if not has_data_pkl:
                self.test_dialogs = RawData._get_dialogs(TEST_MODE,
                                                         self.dialog_vocab,
                                                         self.image_fea)
                print('Save testdata!')
                # Save common data to a .pkl file.
                save_pkl(self.test_dialogs, 'test_dialogs',
                         DatasetConfig.test_raw_data_file)
        print('err_list:', err_list)
        print('err_list_len:', len(err_list))
        print('err_list_value:', err_list_value)
        print('err_list_value_len:', len(err_list_value))
        # exit()

    @staticmethod
    def check_consistency(mode: int) -> None:
        """Check data consistency.

        Args:
            mode (int): Mode (TRAIN_MODE, VALID_MODE, TEST_MODE).
                        e.g. Mode=TRAIN_MODE | VALID_MODE

        Raises:
            ValueError: Raises ValueError if data is inconsistent.

        """

        # Check if dialog data exists.
        if mode & TRAIN_MODE:
            # 原始数据---此处为训练数据存放的地址
            has_data_dir = isdir(DatasetConfig.train_dialog_data_directory)
            # 生成之后的数据
            has_data_pkl = isfile(DatasetConfig.train_raw_data_file)

            if not has_data_dir and not has_data_pkl:
                raise ValueError("No training dataset.")

        if mode & VALID_MODE:
            has_data_dir = isdir(DatasetConfig.valid_dialog_data_directory)
            has_data_pkl = isfile(DatasetConfig.valid_raw_data_file)

            if not has_data_dir and not has_data_pkl:
                raise ValueError("No validation dataset.")

        if mode & TEST_MODE:
            has_data_dir = isdir(DatasetConfig.test_dialog_data_directory)
            has_data_pkl = isfile(DatasetConfig.test_raw_data_file)

            if not has_data_dir and not has_data_pkl:
                raise ValueError("No testing dataset.")

        # Check if there's no common data but data of specific mode exists.
        # 如果common文件不存在
        if not isfile(DatasetConfig.common_raw_data_file):
            consistent = True
            if mode & TRAIN_MODE and isfile(DatasetConfig.train_raw_data_file):
                consistent = False
            if mode & VALID_MODE and isfile(DatasetConfig.valid_raw_data_file):
                consistent = False
            if mode & TEST_MODE and isfile(DatasetConfig.test_raw_data_file):
                consistent = False
            if not consistent:
                raise ValueError("Extracted common data doesn't exist"
                                 " but extracted specific mode data exists.") # 明白了，common文件中保存了所有common文件，下边的train-test-valid是单独的specific文件
                                                                              # 对应地，common文件的保存，里面也应该保存了所有common相关信息

            # Train and valid data is necessary to get common data.
            has_train_dir = isdir(DatasetConfig.train_dialog_data_directory)
            has_valid_dir = isdir(DatasetConfig.valid_dialog_data_directory)
            if not has_train_dir or not has_valid_dir:
                raise ValueError(
                    "Expected train and valid dialog data to extract vocab.")

    def read_extracted_data(self) -> None:
        """ Read existed data.

        Data consists of common data and specific mode data.

        """

        # Common data
        if isfile(DatasetConfig.common_raw_data_file):
            # common_data是属于前边定义的CommonData类型的对象，有这几个value
            common_data: CommonData = load_pkl(
                DatasetConfig.common_raw_data_file)
            self.dialog_vocab = common_data.dialog_vocab
            self.glove = common_data.glove
            self.image_fea = common_data.image_fea
            # self.image_paths = common_data.image_paths

        # Specific mode data
        if self.mode & TRAIN_MODE and isfile(DatasetConfig.train_raw_data_file):
            train_data = load_pkl(DatasetConfig.train_raw_data_file)
            self.train_dialogs = train_data
        if self.mode & VALID_MODE and isfile(DatasetConfig.valid_raw_data_file):
            valid_data = load_pkl(DatasetConfig.valid_raw_data_file)
            self.valid_dialogs = valid_data
        if self.mode & TEST_MODE and isfile(DatasetConfig.test_raw_data_file):
            test_data = load_pkl(DatasetConfig.test_raw_data_file)
            self.test_dialogs = test_data


    @staticmethod
    def _get_common_data() -> CommonData:
        """Get common data.

        Returns:
            * CommonData: Common data.

        """
        # word_index--4892
        dialog_vocab = RawData._get_dialog_vocab()  # 字典格式：word--index(此处的index是单词在词库中的顺序)
        # print('dialog_vocab_len: ', len(dialog_vocab.keys()))
        # word_repre--4892
        glove = RawData._get_glove(dialog_vocab)  #列表格式，第i个元素，对应vocab中的第i个单词的表征 # 字典格式：index---word的表征(此处的index对应于)
        # print('glove_len: ', len(glove))
        # print('glove!')
        # exit()
        # 113560张图片---但是列表中有113561---第一个是空向量的表征
        # image_url_id---图片名字到index的映射---061d23abd8c4a15636eedb5cad764b6e.jpg': 113556
        # image_fea---按照顺序排放图片特征
        # img_index, img_repre---113561
        image_dict, image_fea = RawData._get_images()   # image_url_id(字典格式: url--len)；image_paths(列表格式：图片名字)
        # print('image_dict_len: ', len(image_dict.keys()))
        # print('image_fea_len: ', len(image_fea))
        # 合并word和img---len:118453
        dialog_incor = copy.deepcopy(dialog_vocab)
        dialog_repre = copy.deepcopy(glove)
        for each_img in image_dict.keys():
            img_index = image_dict[each_img]
            dialog_incor[each_img] = len(dialog_incor)
            each_img_repre = image_fea[img_index]
            dialog_repre.append(each_img_repre)
        # print('dialog_incor_len: ', len(dialog_incor))
        # print('dialog_repre_len: ', len(dialog_repre))
        # exit()
        # return CommonData(dialog_vocab=dialog_vocab, glove=glove, image_url_id=image_url_id, image_fea=image_fea)
        # dialog_vocab：只是单词
        return CommonData(dialog_vocab=dialog_incor, glove=glove, image_fea=image_fea)


    @staticmethod
    def _get_dialog_vocab() -> Dict[str, int]:
        """Get dialog vocabulary.

        Returns:
            Dict[str, int]

        """
        # word_freq_cnt是Counter对象 
        # train和valid共享一个Counter对象，所以是基于train和valid构建的一个词库
        word_freq_cnt = Counter()
        # 没有接收返回值
        #基于train和validate产生词库
        RawData._process_dialog_dir(DatasetConfig.train_dialog_data_directory,
                                    word_freq_cnt=word_freq_cnt)
        RawData._process_dialog_dir(DatasetConfig.valid_dialog_data_directory,
                                    word_freq_cnt=word_freq_cnt)
        # SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]
        # 浅拷贝，这里的“浅”指的是相对于嵌入部分“浅”，嵌入部分指的是子对象，由于未对子对象进行拷贝，所以原始数据改变时子对象会改变
        # 是基于special_tokens产生的，所以前几个元素都是special_token的元素
        words = copy.copy(SPECIAL_TOKENS)
        # dialog_text_cutoff = 4只保留出现次数大于4的单词---注意，此处词库的构建是使用的train和validate的单词，没有使用test单词
        words += [word for word, freq in word_freq_cnt.most_common()
                  if freq >= DatasetConfig.dialog_text_cutoff]
        # wid指的单词在词库中的顺序，即为index--第一个词库中的单词是1，第二个是2，第三个是3....
        vocab: Dict[str, int] = {word: wid for wid, word in enumerate(words)}
        # 加入图片

        # print(vocab)
        # print(len(vocab))
        # exit()
        return vocab

    @staticmethod
    def _get_glove(vocab: Dict[str, int]) -> List[Optional[List[float]]]:
        """Get GloVe (Global Vectors for Word Representation)
        Args:
            vocab (Dict[str, int]): Vocabulary.

        Returns:
            List[Optional[List[float]]]: Extracted GloVe, which each element
            in the list is either a float vector or None (no such word in
            GloVe file). Element of index i is the GloVe of ith word (
            corresponding to vocab).
        """
        # Read raw Glove file.
        print('Reading GloVe file {}...'.format(DatasetConfig.glove_file))
        with open(DatasetConfig.glove_file, 'r') as file:
            # 次数的str是什么呢
            raw_glove: Dict[str, List[float]] = {}
            for line in file:
                line = line.strip().split(' ')
                if line:
                    raw_glove[line[0]] = list(map(float, line[1:]))

        # Extract needed vectors.
        glove: List[Optional[List[float]]] = [None] * len(vocab)
        for word, idx in vocab.items():
            # 为什么要让这个idx大于它的长度呢---明白了，因为词库是基于原始special_tokens的列表产生的，所以大于它是指的遍历真正的词，而不是前边几个的特殊符号，前边几个的特殊符号值为空
            if idx >= len(SPECIAL_TOKENS):
                # idx指的是word的index(此处的index其实是单词在vocab中的顺序)--对应单词的表征
                # 如果单词在词库中，则返回对应的表示，不然的话，则返回空
                glove[idx] = raw_glove.get(word, None)
        return glove



    @staticmethod
    def _process_dialog_dir(dialog_dir: str, vocab: Dict[str, int] = None,
                            image_url_id: Dict[str, int] = None,
                            word_freq_cnt: Counter = None) -> List[Dialog]:
        """Process dialog directory.

        Args:
            dialog_dir (str): Dialog directory.
            vocab (Dict[str, int], optional): Vocabulary.
            image_url_id (Dict[str, int], optional): Image URL to index.
            word_freq_cnt (Counter, optional): Word frequency counter.

        Note:
            * (word_freq_cnt is not None) or (vocab is not None and
              image_url_id is not None) == True
            * word_freq_cnt will be updated if it's not None

        Returns:
            List[Dialog]: Extracted dialogs. Empty list if
            word_freq_cnt is not None.

        """

        # Check arguments.
        # assert作用是如果它的条件返回错误，则终止程序执行。----从而可以判断是构建common数据还是基于common数据构建对话的表征
        assert (word_freq_cnt is not None) or (vocab is not None and
                                               image_url_id is not None)
        # word_freq_cnt: Counter()
        # get_vocab: True
        # print('word_freq_cnt: ', word_freq_cnt)
        # 若是空的，则get_vocab是False
        get_vocab: bool = word_freq_cnt is not None
        # print('get_vocab: ', get_vocab)
        # exit()
        # 数据存放的路径
        # 所以这个地方要跑两次
        print('Processing dialog directory {}...'.format(dialog_dir))
        files = listdir(dialog_dir)
        # dialogue_vec_path = '/home/share/chenxiaolin/MultimodalDialogSystem/Dataset/MMConv/dataset/text_vec/'
        # 每条utterance的表征是文本和图片级联在一起
        dialogue_vec_path = '/home/share/chenxiaolin/MultimodalDialogSystem/Dataset/MMConv/dataset/text_img_vec/'
        # dialogue_domain_path = '/home/share/chenxiaolin/MultimodalDialogSystem/Dataset/MMConv/dataset/text_domain/'
        # Useless if word_freq_cnt is not None.
        dialogs: List[Dialog] = []
        # 遍历每一个文件，构建词库
        for file_idx, file in enumerate(files):
            print(file)
            if file.endswith('.json'):
                full_path = join(dialog_dir, file)
                # vec_file
                vec_path = dialogue_vec_path + file.replace('.json', '.npy')
                # domain_path = dialogue_domain_path + file.replace('.json', '.npy')
                # Print current progress.
                # 每处理100个文件，便输出一次
                if (file_idx + 1) % DIALOG_PROC_PRINT_FREQ == 0:
                    print('Processing dialog directory: {}/{}'.format(
                        file_idx + 1, len(files)))
                # Load JSON.---加载json文件
                try:
                    # 原始内容
                    dialog_json1 = json.load(open(full_path))
                    dialog_json = dialog_json1.get('dialogue')
                    # 每一条utterancey预训练表征
                    dialog_npy11 = np.load(vec_path, allow_pickle=True).tolist()
                    dialog_npy1 = dialog_npy11.get('dialogue')
                    # 每个对话的domain信息
                    # dialog_npy22 = np.load(domain_path, allow_pickle=True).tolist()
                    # dialog_npy2 = dialog_npy22
                except json.decoder.JSONDecodeError:
                    continue

                # Extract useful information.
                dialog = []
                # 遍历该对话中的每一轮
                # for utter_dict in dialog_json:
                for i in range(len(dialog_json)):
                    utter_dict = dialog_json[i]
                    # utterance_vec
                    utter_npy1 = dialog_npy1[i]
                    # domain
                    # utter_npy2 = dialog_npy2[i]
                    
                    # 现在get_vocab是True
                    agent_utter:str = utter_dict.get('agent').get('transcript')
                    user_utter:str = utter_dict.get('user').get('transcript')
                    if not get_vocab:
                        utter = RawData._get_utter_from_dict(vocab,
                                                             image_url_id,
                                                             utter_dict, utter_npy1)
                        # dialog.append(utter)
                        dialog.extend(utter)
                        # print(dialog)
                        # exit()
                    else:
                        # 学习到了取连环字典的值
                        # text: str = utter_dict.get('utterance').get('nlg')
                        if agent_utter is None:
                            agent_utter = ''
                        if user_utter is None:
                            user_utter = ''                            
                        # 分词
                        agent_words: List[str] = word_tokenize(agent_utter)
                        # 把单词都换成小写
                        agent_words = [word.lower() for word in agent_words]
                        user_words: List[str] = word_tokenize(user_utter)
                        # 把单词都换成小写
                        user_words = [word.lower() for word in user_words]                        
                        # 
                        if get_vocab:
                            # word_freq_cnt是Counter对象
                            # update后的参数可以是：可迭代对象或者映射操作原理：如果要更新的关键字已存在，则对它的值进行求和；如果不存在，则添加
                            word_freq_cnt.update(agent_words)
                            word_freq_cnt.update(user_words)

                if not get_vocab:
                    # 每一个对话是一个列表，里面的每条utterance使用一个括号表示
                    dialogs.append(dialog)
            # exit()
        return dialogs


    @staticmethod
    def _get_utter_from_dict(vocab: Dict[str, int],
                             image_url_id: Dict[str, int],
                             utter_dict: dict, utter_npy:dict) -> Utterance:
        """Extract Utterance object from JSON dict.

        Args:
            vocab (Dict[str, int]): Vocabulary.
            image_url_id (Dict[str, int]): Image URL to index.
            utter_dict (dict): JSON dict.

        Returns:
            Utterance: Extracted Utterance.

        """
        # phrase
        agent_phrase1 = utter_dict.get('agent').get('phrase')
        user_phrase1 = utter_dict.get('user').get('phrase')

        agent_phrase = extract_phrase(agent_phrase1)
        user_phrase = extract_phrase(user_phrase1)

        # agent_text: List[int] = [vocab.get(word.lower(), UNK_ID) for word in agent_words]

        # venue_infor
        # 不仅仅取dialog_act和turn_label，还考虑到bstate中的name
        agent_act = utter_dict.get('agent').get('dialog_act')
        user_act = utter_dict.get('user').get('turn_label')
        agent_venue, agent_venue_name = extract_venue(agent_act)
        user_venue, user_venue_name = extract_venue(user_act)
        # print('agent_venue_1: ', agent_venue)
        # print('user_venue_1: ', user_venue)

        turn_venue = utter_dict.get('bstate')
        turn_venue_list, turn_venue_name = extract_venue(turn_venue)
        for each_venue in turn_venue_list:
            if each_venue not in agent_venue:
                agent_venue.extend([each_venue])
            if each_venue not in user_venue:
                user_venue.extend([each_venue])

        for each_venue_name in turn_venue_name:
            if each_venue_name not in agent_venue_name:
                agent_venue_name.extend([each_venue_name])
            if each_venue_name not in user_venue_name:
                user_venue_name.extend([each_venue_name])

        agent_domain:list = extract_domain(agent_venue_name)
        user_domain:list = extract_domain(user_venue_name)

        # agent_domain:int = domain_dict[utter_npy2.get('agent').get('domain')]
        # user_domain:int = domain_dict[utter_npy2.get('user').get('domain')]

        # print('agent_venue_2: ', agent_venue)
        # print('user_venue_2: ', user_venue)

        # domain_dict = {'food':1, 'hotel':2, 'nightlife':3, 'mall':4, 'sightseeing':5}
        # 好奇为什么使用get方法，而不是直接使用键值呢
        # utter = utter_dict.get('utterance')
        agent_utter:str = utter_dict.get('agent').get('transcript')
        user_utter:str = utter_dict.get('user').get('transcript')

        agent_img_origin:list = utter_dict.get('agent').get('imgs') 
        user_img_origin:list = utter_dict.get('user').get('imgs')

        agent_vec:list = utter_npy.get('agent').get('trans_vec')
        user_vec:list = utter_npy.get('user').get('trans_vec')


        agent_speaker: str = SYS_SPEAKER
        user_speaker: str = USER_SPEAKER

        _utter_type: str = 'greeting'  # unknown
        # _text: str = utter.get('nlg')
        agent_pos_images: List[str] = utter_dict.get('agent').get('imgs')
        user_pos_images: List[str] = utter_dict.get('user').get('imgs')

        _neg_images: List[str] = []

        # Some attributes may be empty.
        if agent_utter is None:
            agent_utter = ""
        if user_utter is None:
            user_utter = ""            
        # if _utter_type is None:
        #     _utter_type = ""
        if agent_pos_images is None:
            agent_pos_images = []
        if user_pos_images is None:
            user_pos_images = []

        if agent_vec != []:
            if agent_vec is None:
                # agent_vec = [[0]*768]
                agent_vec = [[0]*2816]

            else:
                agent_vec = agent_vec.tolist()
        else:
            # agent_vec = [[0]*768]
            agent_vec = [[0]*2816]
        if user_vec != []:
            # 此处关于None的判断，要使用is，不可以使用==
            if user_vec is None:
                # user_vec = [[0]*768]
                user_vec = [[0]*2816]

            else:
                user_vec = user_vec.tolist()  
        else:
            # user_vec = [[0]*768]
            user_vec = [[0]*2816]

        # Convert speaker into an integer.--记录是否是speak和system是很重要的，因为我们是想预测system说的话
        # speaker: int = -1
        # if _speaker == 'user':
        #     speaker = USER_SPEAKER
        # elif _speaker == 'system':
        #     speaker = SYS_SPEAKER
        # assert speaker != -1

        # Convert utterance type into an integer.
        # get方法若有对应内容，则返回内容，不然的话，若找不到内容，则返回0
        utter_type: int = DatasetConfig.utterance_type_dict.get(_utter_type, 0)
        # We don't care the type of system response.
        # if speaker == SYS_SPEAKER:
        #     utter_type = 0
        # print('_text: ', _text)
        # Convert text into a list of integers.
        # words: List[str] = word_tokenize(_text)
        agent_words: List[str] = word_tokenize(agent_utter)
        user_words: List[str] = word_tokenize(user_utter)

        # 获取单词对应的id，不然的话，返回UNK_ID
        agent_text: List[int] = [vocab.get(word.lower(), UNK_ID) for word in agent_words]
        user_text: List[int] = [vocab.get(word.lower(), UNK_ID) for word in user_words]
        # print('agent_text: ', agent_text)
        # print('user_text: ', user_text)

        # agent_img_origin, user_img_origin--img的index--拼在text后边
        agent_img =  [vocab.get(img, UNK_Img_ID) for img in agent_img_origin] # vocab.get 在词表vocab中有word这个单词，那么就取出它的id；如果没有，就去除UNK（未知词）对应的id
        user_img =  [vocab.get(img, UNK_Img_ID) for img in user_img_origin]
        # print('agent_img: ', agent_img)
        # print('user_img: ', user_img)

        # 包含文本和图片的text_index---所以文本中是包含了文本和图片两种
        agent_text.extend(agent_img)
        user_text.extend(user_img)
        # print('agent_text: ', agent_text)
        # print('user_text: ', user_text)

        # Images--注释掉
        # 获取image对应的index
        agent_pos_images: List[int] = agent_img
        user_pos_images: List[int] = user_img
        neg_images: List[int] = []       
        # agent_pos_images: List[int] = [image_url_id.get(img, 0)
        #                          for img in agent_pos_images]
        # user_pos_images: List[int] = [image_url_id.get(img, 0)
        #                          for img in user_pos_images]
        # neg_images: List[int] = [image_url_id.get(img, 0)
        #                          for img in _neg_images]
        # 把utterance合并到一起
        agent_utter = Utterance(agent_speaker, utter_type, agent_text, agent_vec, agent_phrase, agent_domain, agent_venue, agent_pos_images, neg_images)
        user_utter = Utterance(user_speaker, utter_type, user_text, user_vec, user_phrase, user_domain, user_venue, user_pos_images, neg_images)
        utter = [agent_utter, user_utter]
        # print('agent_utter.agent_text:', agent_utter.text)
        # print('agent_utter.agent_vec:', np.shape(agent_utter.text_vec))
        # print('user_utter.user_text:', user_utter.text)
        # print('user_utter.user_vec:',  np.shape(user_utter.text_vec))
        print('agent_utter.agent_phrase:', agent_utter.phrase)
        print('user_utter.agent_phrase:', user_utter.phrase)

        # if len(agent_img_origin)>0 or len(user_img_origin)>0:
        #     exit()

        return utter

    @staticmethod
    def _get_images() -> Tuple[Dict[str, int], List[str]]:
        """Get images (URL and filenames of local images mapping).

        URL -> Path => URL -> index & index -> Path

        Returns:
            Dict[str, int]: Image URL to index.
            List[str]: Index to the filename of the local image.

        """
        # image_url_id---图片名字到index的映射
        # image_fea---按照顺序排放图片特征
        image_file_list = os.listdir(DatasetConfig.image_data_directory)
        image_url_id: Dict[str, int] = {'': 0}
        # 空向量的表征---对应字典中第一个图片的index
        image_fea = [[0]*2048]
        for img in image_file_list:
            # img: img_id
            image_url_id[img] = len(image_url_id)
            path_img_fea = DatasetConfig.image_fea_directory + img.replace('.jpg','.npy')
            # print(path_img_fea)
            img_fea_vec = np.load(path_img_fea).tolist()
            # image_url_id是字典--key是图片名字，value是index
            # image_fea是列表--里面是按照idx排序的表征
            image_fea.append(img_fea_vec)

        return image_url_id, image_fea
            # print(str(len(image_file_list))+'-----'+str(len(image_url_id)))

            # print(type(img_fea_vec))            
            # print(np.shape(img_fea_vec))            
            # exit()

        # image_fea_file_list = os.listdir(DatasetConfig.image_fea_directory)
        # print(image_fea_file_list)
        # print('image:',len(image_file_list))
        # print('image:',len(image_fea_file_list))
        # exit()
        # # Get URL to filename mapping dict.
        # with open(DatasetConfig.url2img, 'r') as file:
        #     url_image_pairs: List[List[str]] = [line.strip().split(' ')
        #                                         for line in file.readlines()]
        # # 列表格式，其中每个元素是(A,B)
        # url_image_pairs: List[Tuple[str, str]] = [(p[0], p[1])
        #                                           for p in url_image_pairs]
        # # 这个具体是什么格式呢，什么是key，什么是value呢---图片url:图片名
        # url2img: Dict[str, str] = dict(url_image_pairs)

        # # Divided it into two steps.
        # # URL -> Path => URL -> index & index -> Pathtrain
        # # Element of index 0 should be empty image.
        # image_url_id: Dict[str, int] = {'': 0}
        # image_paths: List[str] = ['']

        # for url, img in url2img.items():
        #     # url对应的是url_id的长度？
        #     image_url_id[url] = len(image_url_id)
        #     # 图片名字
        #     image_paths.append(img)
        # return image_url_id, image_paths

    @staticmethod
    def _get_dialogs(mode: int,
                     vocab: Dict[str, int],
                     image_url_id: Dict[str, int]) -> List[Dialog]:
        """Get mode specific dialogs.

        Args:
            mode (int): TRAIN_MODE / VALID_MODE / TEST_MODE.
            vocab (Dict[str, int]): Vocabulary.
            image_url_id (Dict[str, int]): Image repre # Image URL to index.


        Returns:
            List[Dialog]: Extracted dialogs of specific mode.

        Raises:
            ValueError: Raises if mode is neither TRAIN_MODE, nor VALID_MODE,
                        nor TEST_MODE.

        """

        if mode == TRAIN_MODE:
            return RawData._process_dialog_dir(
                DatasetConfig.train_dialog_data_directory, vocab, image_url_id)
        # exit()
        if mode == VALID_MODE:
            return RawData._process_dialog_dir(
                DatasetConfig.valid_dialog_data_directory, vocab, image_url_id)
        if mode == TEST_MODE:
            return RawData._process_dialog_dir(
                DatasetConfig.test_dialog_data_directory, vocab, image_url_id)
        raise ValueError('Illegal mode.')
