"""Tidy data module."""
import copy
from os.path import isfile, join
from typing import List, Dict
import numpy as np
from config import DatasetConfig
from constant import INTENTION_TASK, TEXT_TASK, RECOMMEND_TASK, KNOWLEDGE_TASK
from constant import KNOWLEDGE_ATTRIBUTE_SUBTASK
from constant import KNOWLEDGE_CELEBRITY_SUBTASK
from constant import KNOWLEDGE_STYLETIP_SUBTASK
from constant import TRAIN_MODE, VALID_MODE, TEST_MODE
from constant import USER_SPEAKER, SYS_SPEAKER
from dataset import RawData
from dataset.model import TidyUtterance, Utterance
from util import save_pkl, get_product_path, load_pkl
from constant import TASK_STR, MODE_STR

Dialog = List[Utterance]
TidyDialog = List[TidyUtterance]
# global number_context
# number_context = 0

def generate_tidy_data_file(raw_data: RawData, task: int, mode: int):
    """Generate tidy data file.

    Args:
        raw_data (RawData): Raw data.
        task (int): A single task.
        mode (int): A single mode.

    """

    # If item file already exists, then return and print a warning
    item_file_name: str = DatasetConfig.get_dialog_filename(task, mode)
    if isfile(item_file_name):
        train_tidy_data = load_pkl(DatasetConfig.text_train_dialog_file)
        print(type(train_tidy_data))
        print(np.shape(train_tidy_data))
        print('Warning: Tidy data file {} exists.'.format(item_file_name))
        return

    # Get raw data dialogs according to its mode.
    dialogs: List[Dialog] = None
    if mode == TRAIN_MODE:
        # (3500,)---3500个对话;每个对话包含多个utterance;每个utterance使用5个元素决定--(speaker, utter_type, text, pos_images, neg_images)
        # 更新成vec和domain信息--便有7个了
        dialogs = raw_data.train_dialogs
        print('train_len(dialogs):', len(dialogs))

    if mode == VALID_MODE:
        dialogs = raw_data.valid_dialogs
        print('VALID_len(dialogs):', len(dialogs))
    
    if mode == TEST_MODE:
        dialogs = raw_data.test_dialogs
        print('TEST_len(dialogs):', len(dialogs))
    assert dialogs is not None

    if task & KNOWLEDGE_TASK:
        ordinal_number = {raw_data.dialog_vocab[key]: value for key, value in
                          DatasetConfig.ordinal_number.items()}
    # exit()
    # tidy_dialogs: List[TidyDialog] = []
    tidy_dialogs: index_List[TidyDialog] = {}
    utter_len_text = []
    utter_len_img= []

    # 对于每个对话分别采取操作----遍历每一个对话
    for item_idx, dialog in enumerate(dialogs):
        # Get items according to different TASKS.
        if task == INTENTION_TASK:
            # Standardize dialog first.
            std_dialog: Dialog = standardized_dialog(dialog)
            tidy_dialogs.extend(get_intention_task_items(std_dialog))
        elif task == TEXT_TASK:
            # 构建完context内容之后的数据
            # dialog的长度是14，其中符合text的是6，所以，构建完的数据shape是(6,3)
            # 所以它的作用是：1)筛选出符合条件的utterance(比如type是text)，2)对于每个符合条件的utterance，构建context--2个。
            # 从0开始
            num_context = len(tidy_dialogs.keys())
            print('num_context: ', num_context)
            # exit()
            dict_return = get_text_task_items(dialog, num_context)
            # text_len, img_len = get_text_task_items(dialog, num_context)
            # utter_len_text.extend(text_len)
            # utter_len_img.extend(img_len)

            # tidy_dialogs.extend(get_text_task_items(dialog, num_context))
            tidy_dialogs.update(dict_return)
            # tidy_dialogs.extend(dict_return)
            # print(tidy_dialogs)
    # path_save_text = '/home/share/chenxiaolin/MultimodalDialogSystem/Model/dataset_dump/'+MODE_STR[mode]+'_utter_text_len.npy'
    # path_save_img = '/home/share/chenxiaolin/MultimodalDialogSystem/Model/dataset_dump/'+MODE_STR[mode]+'_utter_img_len.npy'
    # np.save(path_save_text, utter_len_text)
    # np.save(path_save_img, utter_len_img)
    # text_len_mean = np.mean(utter_len_text)
    # img_len_mean = np.mean(utter_len_img)
    # print('mode:', MODE_STR[mode])
    # print('text_len_mean:', text_len_mean)
    # print('img_len_mean:', img_len_mean)
    
    #==注释掉# print(len(tidy_dialogs.keys()))
    # exit()
    # Save as pickle file.---注释掉
    for context_index in tidy_dialogs.keys():
        print('train_context_index_final_check: ', context_index)
        context_content = tidy_dialogs[context_index]
        if len(context_content) != 11:
            print(len(context_content))
            exit()
    # exit()
    save_pkl(tidy_dialogs, 'tidy_dialogs', item_file_name)


def standardized_dialog(dialog: Dialog) -> Dialog:
    """Standardized raw dialog.

    Args:
        dialog (Dialog): Raw dialog.

    Returns:
        Dialog: Standard dialog.

    """
    std_dialog: Dialog = []
    for utter in dialog:
        if not std_dialog and utter.speaker != USER_SPEAKER:
            std_dialog.append(Utterance(USER_SPEAKER, -1, [], [], []))
        if not std_dialog or utter.speaker != std_dialog[-1].speaker:
            std_dialog.append(utter)
        else:
            std_dialog[-1].utter_type = utter.utter_type
            std_dialog[-1].text += utter.text
            std_dialog[-1].pos_images += utter.pos_images
            std_dialog[-1].neg_images += utter.neg_images
    return std_dialog


def get_init_pad_utters() -> List[TidyUtterance]:
    """Get initial padding utterances.

    Returns:
        List[TidyUtterance]

    """
    utters: List[TidyUtterance] = []
    # dialog_context_size=2，则i的取值是0,1
    for i in range(DatasetConfig.dialog_context_size):
        # dialog_context_size = 2
        # 取余，i=0， 是user---和我们的数据集不一样---MMCONV是从agent开始的----不不，是一样的，这里是指的，当前输出是agent输出，则当前输出的前两条---第一条是agent,第二条是user
        if (DatasetConfig.dialog_context_size - i - 1) % 2 == 0:
            speaker = SYS_SPEAKER
        else:
            speaker = USER_SPEAKER
        # Utterance对象的格式是str((self.speaker, self.utter_type, self.text, self.pos_images, self.neg_images))
        # 包含5个属性，和raw_data构建的一样
        # text_vec = [0]*768
        text_vec = [0]*2816
        # domain表示为0--代表补全的domain
        # utter = Utterance(speaker, -1, [], text_vec, 0, [], [])
        # utter = Utterance(speaker, -1, [], text_vec, -1, 0, [], [])
        # utter = Utterance(speaker, -1, [], text_vec, -1, [], [], [])
        # utter = Utterance(speaker, -1, [], text_vec, [-1], [], [], [])
        utter = Utterance(speaker, -1, [], text_vec, [], [-1], [], [], [])



        # (self.speaker, self.utter_type, self.text, self.text_vec, self.domain, self.venue, self.pos_images, self.neg_images)
        # print(utter)
        # utters中包含了两个utterance，但是utterance的text, pos和neg是空的---因为context_size是2
        # TidyUtterance对象的格式是str((self.utter_type, self.text, self.text_len, self.pos_images, self.pos_images_num,self.neg_images, self.neg_images_num))
        # 包含7个属性,text中包含特殊字符：SOS_ID = 0, EOS_ID = 1, UNK_ID = 2, PAD_ID = 3
        # neg_images设置是4张，所以这里是4
        utters.append(TidyUtterance(utter))
        # print(utters)
        # exit()
    return utters


def get_intention_task_items(dialog: Dialog) -> List[TidyDialog]:
    """Get items for intention task from a single dialog.

    Args:
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()

    for utter in dialog:
        if utter.speaker == USER_SPEAKER:
            utterances.append(TidyUtterance(utter))
        if utter.speaker == SYS_SPEAKER:
            utterances.append(TidyUtterance(utter))
            utterances = utterances[-(DatasetConfig.dialog_context_size + 1):]
            dialogs.append(copy.copy(utterances))
    return dialogs


def get_text_task_items(dialog: Dialog, number_context:int) -> List[TidyDialog]:
    # 此处的dialog是指的原始对话使用单词index表征
    """Get items for text task from a single dialog.

    Args:
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """
    utter_text_len = []
    utter_img_len = []

    dialogs: List[TidyDialog] = {}
    # context--每条utterance有7个属性--tinyutterance对象
    utterances = get_init_pad_utters()
    utter_vec = []
    # print(utterances)
    # exit()
    utter_type = None
    sys_responses: List[Utterance] = []
    context_size = DatasetConfig.dialog_context_size
    # number = 0
    # number_context = 0
    # print('--------origin--------',dialog)
    # 每条utter有5个属性
    for utter in dialog:
        # 统计每条utterance的单词数目和image的数目
        # text_list = []
        # img_list = []
        utter_text = utter.text
        # for each_text in utter_text:
        #     if each_text <4892:
        #         text_list.extend([each_text])
        #     else:
        #         img_list.extend([each_text])
        # utter_text_len.extend([len(text_list)]) # utter_text_len, utter_img_len
        # utter_img_len.extend([len(img_list)])

        if utter.speaker == USER_SPEAKER:
            # 查看之前是否有积累的是system发出的内容,但是不符合task的内容,如有这种情况,则把这些内容放入context中
            # 放入当前user发布的内容当作context
            # The first utterance of three consecutive system responses must be
            # a simple response, and after getting this simple response dialog.
            # The other two responses should be in the candidate context.
            # if len(sys_responses) == 3:
            # if len(sys_responses) == 11:
            #     for idx, response in enumerate(sys_responses):
            #         # 每条回复依次放到utterances中--11条都放进去
            #         utterances.append(TidyUtterance(response))
            #         # 如果sys连续积累的超过3条,那么把积累的第一条当作taget,其余的放到context中
            #         if idx == 0:
            #             # 如果是第一条内容,则直接取这个内容和之前的放到context中--
            #             utterances = utterances[-(context_size + 1):]
            #             # dialogs.append(copy.copy(utterances))
            #             dialogs[number_context] = copy.copy(utterances)
            #             # global number_context
            #             number_context = number_context + 1
            # elif sys_responses:
            #     # If there are no three consecutive system responses, then just
            #     # append them to the candidate context.
            #     # 直接把这个放到历史对话中
            #     for response in sys_responses:
            #         # ((self.speaker, self.utter_type, self.text, self.pos_images, self.neg_images))
            #         # 如果sys_responses有值,但是不为3,则先把他放到context中
            #         utterances.append(TidyUtterance(response))
            # 清空sys_response---只有连续出现sys_utter,才会积累,不然的话,会清空
            # sys_responses = []
            # 若utter.speaker是user，且是第一个，则直接把这一个放到context中
            # user说的utterance肯定会是context
            # 转化成7个属性--没有speaker属性了
            # 只保留不为空的utter
            if len(utter_text)!= 0:
                utterances.append(TidyUtterance(utter))
            # else:
            #     print('utter: ', utter)
            #     print('utter_text: ', utter_text)
            #     print('len(utter_text):', len(utter_text))                
            #     # print(utterances)
            #     exit()
            # print('utterances:', utterances)
            # print('utterances:', len(utterances))
            # exit()
            utter_type = utter.utter_type
        elif utter.speaker == SYS_SPEAKER:
            # If the type of last user utterance is in utterance_text_types
            # then it's also a simple response
            # 因为这个是预测文本生成的，所以，只存放了属于文本类型的内容
            # 判断是否是text类型的---utter_type全部都是greeting
            # if utter.utter_type in DatasetConfig.utterance_text_types or \
            #         utter_type in DatasetConfig.utterance_text_recommend_types:
                # number = number + 1
                # 把这条内容放到了context列表中
            # 只保留文本不为空的utter
            # utter_text = utter.text
            # 只保留不为空的utter
            if len(utter_text)!= 0:
                utterances.append(TidyUtterance(utter))
            # 然后取值-放进去一个，则只取后3个---[-3:]
            # [start_index:end_index:step]  end_index:表示结束索引（不包含该索引对应的值）
                # utterances = utterances[-(context_size + 1):]
                utterances1 = utterances[-(context_size + 1):]
                # print('---utterances-----:', len(utterances))
            # 取目标utterance之前的三条
            # 如果system说的utterance满足条件,则直接取前两个utterance和它自己放到内容中---每条utterance是7个属性
            # dialogs.append(copy.copy(utterances))
                # dialogs[number_context] = copy.copy(utterances)
                dialogs[number_context] = utterances1
                # print('---number_context-----:', number_context)
                # if len(dialogs[number_context]) != 11:
                    # exit()
                # print('---dialogs[number_context]-----:', len(dialogs[number_context]))
                # if number_context==10:
                #     exit()
            # else:                
            # global number_context
                # 没有必要加1，因为下次是通过函数传进来
                number_context = number_context + 1
                utter_type = utter.utter_type

            #     print('utter: ', utter)
            #     print('utter_text: ', utter_text)
            #     print('len(utter_text):', len(utter_text))
            #     exit()                
            # else:
            #     # 若不符合条件，则放进这个sys_responses列表中--到user部分统一放进去
            #     sys_responses.append(utter)
        # 以上全部
    # print('number:', str(number))
    # 如果最后是3条system发出的内容结尾(此时就不再经过第一个USER_SPEAKER循环了),则把这三条内容中的第一条当作是target内容,其余的两条不管了
    # if len(sys_responses) == 3:
    # 以下全部
    # if len(sys_responses) == 11:
    #     utterances.append(TidyUtterance(sys_responses[0]))
    #     utterances = utterances[-(context_size + 1):]
    #     # dialogs.append(copy.copy(utterances))
    #     dialogs[number_context] = copy.copy(utterances)
    #     # global number_context
    #     number_context = number_context + 1
    # 以上全部
    # print('-----proceed-----', dialogs)
    # exit()
    return dialogs
    # return utter_text_len, utter_img_len



def get_valid_image(image_paths: List[str],
                    images: List[int]) -> List[int]:
    """ Get valid images in `images`.

    Args:
        image_paths (List[str]): Image paths.
        images (List[int]): Images.

    Returns:
        valid_images (List[int]): Valid images.

    """
    res_images = []
    for image_id in images:
        image_path = join(DatasetConfig.image_data_directory,
                          image_paths[image_id])
        if not isfile(image_path):
            continue
        product_path = get_product_path(image_paths[image_id])
        if not isfile(product_path):
            continue
        res_images.append(image_id)
    return res_images


def get_recommend_task_items(
        image_paths: List[str], dialog: Dialog) -> List[TidyDialog]:
    """Get items for recommend task from a single dialog.

    Args:
        image_paths (List[str]): Image paths.
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()
    utter_type = None
    sys_responses: List[Utterance] = []
    context_size = DatasetConfig.dialog_context_size

    for utter in dialog:
        if utter.speaker == USER_SPEAKER:
            selected_idx = -1
            for idx, response in enumerate(sys_responses):
                pos_images = get_valid_image(image_paths, response.pos_images)
                if pos_images:
                    neg_images = get_valid_image(image_paths,
                                                 response.neg_images)
                    if neg_images:
                        response.pos_images = pos_images
                        response.neg_images = neg_images
                        utterances.append(TidyUtterance(response))
                        utterances = utterances[-(context_size + 1):]
                        dialogs.append(copy.copy(utterances))
                        selected_idx = idx
                        break
            for response in sys_responses[selected_idx + 1:]:
                utterances.append(TidyUtterance(response))
            sys_responses = []
            utterances.append(TidyUtterance(utter))
            utter_type = utter.utter_type
        elif utter.speaker == SYS_SPEAKER:
            if utter_type in DatasetConfig.utterance_recommend_types:
                sys_responses.append(utter)
            else:
                utterances.append(TidyUtterance(utter))

    for response in sys_responses:
        pos_images = get_valid_image(image_paths, response.pos_images)
        if pos_images:
            neg_images = get_valid_image(image_paths, response.neg_images)
            if neg_images:
                response.pos_images = pos_images
                response.neg_images = neg_images
                utterances.append(TidyUtterance(response))
                utterances = utterances[-(context_size + 1):]
                dialogs.append(copy.copy(utterances))
                break

    return dialogs


def get_products(order_words, text, products):
    result = []
    if products:
        for word in text:
            if word in order_words:
                order = order_words[word]
                if order < len(products):
                    product = products[order]
                    result.append(product)
    if not result:
        if len(products) > 0:
            result.append(products[0])
        else:
            result.append(0)
    return result


def get_knowledge_items(dialog: Dialog, ordinal_number: Dict[int, int],
                        task: int) -> List[TidyDialog]:
    """Get items for knowledge task from a single dialog.

    Args:
        dialog (Dialog): Dialog.
        ordinal_number (Dict[int, int]): Ordinal numbers.
        task (int): Task.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """
    expected_utter_types = {}
    if task == KNOWLEDGE_STYLETIP_SUBTASK:
        expected_utter_types = DatasetConfig.utterance_knowledge_styletip_types
    elif task == KNOWLEDGE_ATTRIBUTE_SUBTASK:
        expected_utter_types = DatasetConfig.utterance_knowledge_attribute_types
    elif task == KNOWLEDGE_CELEBRITY_SUBTASK:
        expected_utter_types = DatasetConfig.utterance_knowledge_celebrity_types

    dialogs: List[TidyDialog] = []
    utterances = get_init_pad_utters()
    context_size = DatasetConfig.dialog_context_size
    utter_type = None
    has_shown = False
    products = []
    selected_products = []
    for utter in dialog:
        pos_images = [image for image in utter.pos_images if image > 0]

        if utter.speaker == USER_SPEAKER:
            utterances.append(TidyUtterance(utter))
            selected_products = get_products(ordinal_number, utter.text,
                                             products)
            utter_type = utter.utter_type
        elif utter.speaker == SYS_SPEAKER:
            desc = task == KNOWLEDGE_ATTRIBUTE_SUBTASK and has_shown and \
                   utter_type in DatasetConfig.utterance_recommend_types and \
                   len(utter.text) > 10
            if utter_type in expected_utter_types or desc:
                if desc:
                    selected_products = get_products(ordinal_number, utter.text,
                                                     products)
                utterances = utterances[-context_size:]
                text = copy.deepcopy(utter.text)
                special_utter = Utterance(utter.speaker,
                                          utter.utter_type,
                                          text,
                                          selected_products,
                                          [])
                special_utter = TidyUtterance(special_utter)
                dialogs.append(copy.deepcopy(utterances + [special_utter]))
                utter_type = None
            utterances.append(TidyUtterance(utter))
            has_shown = False
            if pos_images:
                products = pos_images
                if utter_type in DatasetConfig.utterance_recommend_types:
                    has_shown = True
    return dialogs
