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
from lib import encode_context
from lib.eval import text_eval
from lib.loss import text_loss
from lib.test import text_test
from lib.valid import text_valid
from model import TextDecoder, ToHidden
from model import TextEncoder, ImageEncoder, ContextEncoder, HyperConv, UtterFusion, domainHyper, KnowledgeIncor, DynamicCorrelation
import numpy as np
import random
import scipy.sparse as sp
from constant import PAD_ID, EOS_ID
import warnings
warnings.filterwarnings('ignore')
import copy
import subprocess
import sys

# base_dir = '/home/share/chenxiaolin/MultimodalDialogSystem/Model/dataset_dump/'
base_dir = '/home/share/chenxiaolin/MultimodalDialogSystem/Model_data_dump10/dataset_dump/'
# base_dir = '/home/share/chenxiaolin/MultimodalDialogSystem/Model/dataset_dump_2/'
path_global_repre = base_dir + 'Context/global_repre/'


# def get_context_dialog(context):
#     dialog = context.get('dialogues')
#     text_list: List[List[int]] = [utter.text for utter in dialog]
#     text_length_list: List[int] = [utter.text_len for utter in dialog]
#     text_vec_list = [np.squeeze(utter.text_vec).tolist() for utter in dialog]
#     texts = torch.stack(tuple([torch.tensor(text) for text in text_list]))
#     texts_vec =  torch.stack([torch.LongTensor(text_vec) for text_vec in text_vec_list])
#     text_lengths = torch.tensor(text_length_list)
#     utter_type = dialog[-2].utter_type
#     return texts, texts_vec, text_lengths, utter_type

# def construct_hypergraph(subgraphs_G, subgraphs, subgraphs_mapping_w, subgraphs_mapping_u, context_index)
#     col = subgraphs[context_index]['u']  # u是超边--utterance
#     row = subgraphs[context_index]['w']  # i是结点--代表item---word/image
#     data = np.ones(len(col))
#     print('len(subgraphs_mapping_w[context_index]', len(subgraphs_mapping_w[context_index]))
#     sg = sp.coo_matrix((data, (row, col)), shape=(len(subgraphs_mapping_w[context_index]), len(subgraphs_mapping_u[context_index])))
#     # print('Done constructing subgraph', str(context_index))
#     # print(len(subgraphs_mapping_w[context_index]), len(subgraphs_mapping_u[context_index]), len(data))
#     subgraphs_G[context_index] = {}
#     subgraphs_G[context_index]['G'], subgraphs_G[context_index]['E'] = generate_G_from_H(sg)
def extract_graph_index(batch_id, batch_size):
    subgraphs_mapping_list = []
    for i in range(batch_id, batch_id+batch_size):
        subgraphs_mapping_list.append(i)
    return subgraphs_mapping_list

def load_context_data(subgraphs, subgraphs_mapping_w, subgraphs_mapping_u, index_list):
    subgraphs_G = {}
    # context_dir = base_dir + 'Context/' + mode +'/'
    for context_index1 in index_list[0]:
        # context_index = int(context_index1)
        context_index = context_index1
        col = subgraphs[context_index]['w']  # u是超边--utterance
        row = subgraphs[context_index]['u']  # i是结点--代表item---word/image
        data = np.ones(len(col))
        # print('len(subgraphs_mapping_w[context_index]', len(subgraphs_mapping_w[context_index]))
        # 元素全为1--没有进行归一化
        sg = sp.coo_matrix((data, (row, col)), shape=(len(subgraphs_mapping_u[context_index]), len(subgraphs_mapping_w[context_index])))
        # print('Done constructing subgraph', str(context_index))
        # print(len(subgraphs_mapping_w[context_index]), len(subgraphs_mapping_u[context_index]), len(data))
        subgraphs_G[context_index] = {}
        subgraphs_G[context_index]['G'], subgraphs_G[context_index]['E'] = generate_G_from_H(sg)
    return subgraphs_G

def domain_node_shuffle(domain_context):
    batch_list = []
    # domain_node_0 = random.sample(domain_context[0], 1539)
    # domain_node_1 = random.sample(domain_context[1], 340) 
    # domain_node_2 = random.sample(domain_context[2], 253) 
    # domain_node_3 = random.sample(domain_context[3], 324) 
    # domain_node_4 = random.sample(domain_context[4], 726)
    # domain_node_0 = random.sample(domain_context[0], 948)
    # domain_node_1 = random.sample(domain_context[1], 267) 
    # domain_node_2 = random.sample(domain_context[2], 123) 
    # domain_node_3 = random.sample(domain_context[3], 209) 
    # domain_node_4 = random.sample(domain_context[4], 414)    
    # domain_node_0 = random.sample(domain_context[0], 947)
    # domain_node_1 = random.sample(domain_context[1], 265) 
    # domain_node_2 = random.sample(domain_context[2], 122) 
    # domain_node_3 = random.sample(domain_context[3], 209) 
    # domain_node_4 = random.sample(domain_context[4], 414)
    domain_node_0 = random.sample(domain_context[0], 948)
    domain_node_1 = random.sample(domain_context[1], 267) 
    domain_node_2 = random.sample(domain_context[2], 123) 
    domain_node_3 = random.sample(domain_context[3], 209) 
    domain_node_4 = random.sample(domain_context[4], 414)            
    batch_list.extend(domain_node_0)
    batch_list.extend(domain_node_1)
    batch_list.extend(domain_node_2)
    batch_list.extend(domain_node_3)
    batch_list.extend(domain_node_4)
    return batch_list

def domain_row_col(domain_node_dict, context_domain):
    subgraph_domain = []
    domain_list = []
    for domain_node in domain_node_dict.keys():
        domain_node1 = domain_node+'.npy'
        context_domain_list = context_domain[domain_node1]
        if -1 in context_domain_list:
            context_domain_list.remove(-1)
        subgraph_domain.extend(context_domain_list)
        for i in range(len(context_domain_list)):
            domain_list.append(domain_node_dict[domain_node])
    # print('common-subgraph_domain:', len(subgraph_domain))
    # print('common-domain_list_len:', len(domain_list))
    return subgraph_domain, domain_list

def domain_row_col2(subgraph_domain1, domain_list1, domain_node_dict, context_domain):
    # subgraph_domain = []
    # domain_list = []
    for domain_node in domain_node_dict.keys():
        domain_node1 = domain_node+'.npy'
        context_domain_list = context_domain[domain_node1]
        if -1 in context_domain_list:
            context_domain_list.remove(-1)
        subgraph_domain1.extend(context_domain_list)
        for i in range(len(context_domain_list)):
            domain_list1.append(domain_node_dict[domain_node])
    # print('domain_list_len:', len(domain_list))
    # print('subgraph_domain:', len(subgraph_domain))
    return subgraph_domain1, domain_list1


def generate_G_from_H(H):
    # H的维度是节点*边--N*M
    # 边的数目
    n_edge = H.shape[1]
    # the weight of the hyperedge--长度为边数目的一维数组(元素全为1)--M
    W = np.ones(n_edge)
    # the degree of the node
    # 每个节点出现的边的数目--N 
    DV = np.array(H.sum(1)) 
    # print('DV:', np.shape(DV))
    # exit()
    # the degree of the hyperedge---每个边中出现X个节点--M
    DE = np.array(H.sum(0))
    # 归一化--按行占成一维
    # --numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵
    # --对角线元素为这个，其余的为0---M*M---归一化
    invDE2 = sp.diags(np.power(DE, -0.5).flatten())
    # N*N
    DV2 =  sp.diags(np.power(DV, -0.5).flatten())
    # M*M
    W = sp.diags(W)
    # M*N
    HT = H.T

    # DV2是干啥的---感觉这个的G是指的做完部分超图卷积之后的矩阵(相对于完整的超图卷积，还没有乘表示)
    # 左边invDE2是对边进行归一化，右边DV2是对节点归一化---这个操作是针对于HT来说的
    invDE_HT_DV2 = invDE2 * HT * DV2
    # 前边这四项是对H矩阵进行操作
    G = DV2 * H * W * invDE2 * invDE_HT_DV2
    return G, invDE_HT_DV2

def construct_domain_hyper(mode, subgraphs_mapping_list, domain_global_repre, subgraph_domain, domain_list, domain_node_dict):
    subgraph_domain1 = copy.deepcopy(subgraph_domain)
    domain_list1 = copy.deepcopy(domain_list)
    # subgraphs_mapping_list = list(subgraphs_mapping_list)
    # path_domain_context = base_dir + 'Context/domain_context_' + mode + '.npy'
    path_context_domain = base_dir + 'Context/context_domain_' + mode + '.npy'
    path_global_repre_mode = path_global_repre + mode + '/'

    # domain_context = np.load(path_domain_context, allow_pickle=True).tolist()
    context_domain = np.load(path_context_domain, allow_pickle=True).tolist()
    # print('context_domain:', context_domain)
    
    # domain_node_list1 = domain_node_shuffle(domain_context)
    # domain_node_list2 = [each_domain.replace('.npy','') for each_domain in domain_node_list1]
    domain_global_repre2 = []
    # domain_node_list2.extend(list(subgraphs_mapping_list))
    # # print('domain_node_list:', domain_node_list1)
    # domain_node_list = list(set(domain_node_list2))
    # len_inital = len
    domain_node_len_1 = np.shape(domain_global_repre)[0]
    domain_node_dict2 = {}
    # domain_global_repre = []
    # print('subgraphs_mapping_list:', subgraphs_mapping_list)
    # print('subgraphs_mapping_list_type:', type(subgraphs_mapping_list))
    # print('subgraphs_mapping_list_len:', len(subgraphs_mapping_list))

    # for domain_node in subgraphs_mapping_list:
    for node_index in range(len(subgraphs_mapping_list)):
        domain_node = subgraphs_mapping_list[node_index]
        # print('domain_node: ', domain_node) 
        path_each_domain = path_global_repre_mode + domain_node + '.npy'
        each_global_repre = np.load(path_each_domain, allow_pickle=True).tolist()
        domain_node_dict2[domain_node] = len(domain_node_dict) + node_index
        # 只保留非空的
        # if isinstance(each_global_repre, list):
        #     domain_global_repre2.append(each_global_repre)
        #     # print(np.shape(each_global_repre))
        # else:
        #     each_global_repre = [float(0)]*2816
        domain_global_repre2.append(each_global_repre)

    # print('inter-domain_node_dict2:', len(domain_node_dict2))
    # print('domain_global_repre: ', np.shape(domain_global_repre))
    # print('domain_global_repre2: ', np.shape(domain_global_repre2))
    subgraph_domain, domain_list =  domain_row_col2(subgraph_domain1, domain_list1, domain_node_dict2, context_domain)  
    # print('inter-subgraph_domain:', np.shape(subgraph_domain))
    # print('inter-domain_list:', np.shape(domain_list))
    # print('subgraph_domain: ', subgraph_domain)
    # print('domain_list: ', domain_list)

    subgraphs_G_domain = {}
    row_domain = domain_list   
    col_domain = subgraph_domain
    data_domain = np.ones(len(col_domain))
    # domain_node_len = len(domain_node_dict) + len(subgraphs_mapping_list)
    # domain_node_len = np.shape(domain_global_repre)[0]+np.shape(domain_global_repre2)[0]
    domain_node_len = np.shape(domain_global_repre)[0]+len(subgraphs_mapping_list)

    
    # print('data_domain_train:', np.shape(data_domain))
    # print('row_domain_train:', np.shape(row_domain))
    # print('col_domain_train:', np.shape(col_domain))
    # print('domain_node_len_train:', domain_node_len)    
    # print('inter-row_domain:', np.shape(row_domain))
    # print('inter-col_domain:', np.shape(col_domain))
    # print('inter-data_domain:', np.shape(data_domain))
    # print('inter-domain_node_len:', domain_node_len)

    sg_domain = sp.coo_matrix((data_domain, (row_domain, col_domain)), shape=(domain_node_len, 5))
    subgraphs_domain_node, subgraphs_domain_edge = generate_G_from_H(sg_domain)

    # repre
    # domain_global_repre_list =  torch.stack([torch.LongTensor(each_repre) for each_repre in domain_global_repre])
    # domain_global_repre_list =  torch.stack([torch.LongTensor(each_repre, requires_grad=True) for each_repre in domain_global_repre])
    # domain_global_repre_list =  torch.stack([torch.long(torch.tensor(each_repre, requires_grad=True)) for each_repre in domain_global_repre])
    # domain_global_repre_list =  torch.stack([torch.Tensor(each_repre, dtype=torch.long, requires_grad=True) for each_repre in domain_global_repre])
    # domain_global_repre_list1 =  torch.stack([torch.tensor(each_repre.float(), requires_grad=True) for each_repre in domain_global_repre2])
    # domain_global_repre_list1 =  torch.stack([torch.tensor(each_repre, requires_grad=True) for each_repre in domain_global_repre2])
    domain_global_repre_list1 =  torch.stack([torch.LongTensor(each_repre) for each_repre in domain_global_repre2]).float()

    # print('domain_global_repre_list1:', np.shape(domain_global_repre_list1))
    domain_global_repre_list2 = torch.cat((domain_global_repre, domain_global_repre_list1),0)
    # print('domain_global_repre_list2:', np.shape(domain_global_repre_list2))
    # domain_global_repre_list2 = domain_global_repre_list[:domain_node_len_1]
    # texts_vec =  torch.stack([torch.long(torch.tensor(text_vec, requires_grad=True)) for text_vec in text_vec_list])
    # print('domain_global_repre_list:', domain_global_repre_list[0])    
    # print('domain_global_repre_list2:', domain_global_repre_list2[0])
    # print('domain_global_repre_list:', np.shape(domain_global_repre_list))
    # print('domain_global_repre_list2:', np.shape(domain_global_repre_list2))

    # print('domain_global_repre_list:', np.shape(domain_global_repre_list))
    # exit()
    return domain_global_repre_list2, subgraphs_domain_node, subgraphs_domain_edge, domain_node_dict2

# def count_length(texts):
#     texts1 = texts.squeeze()
#     texts_list = []
#     for each_tar in texts:
#         # each_tar = texts[i]
#         for j in range(len(each_tar)):
#             if each_tar[j] == EOS_ID:
#                 len_tar = j+1
#                 texts_list.append([len_tar])
#                 break
#     text_lengths = torch.tensor(texts_list)

#     return  text_lengths

def target_id2word(target, id2word, context_len, num_context, context_len_list):
    # all_tokens = torch.zeros((text_length, batch_size), dtype=torch.long)
    # all_tokens = all_tokens.to(GlobalConfig.device)
    batch_target = []
    # print('target: ', target)
    # print('id2word: ', id2word)
    # print('target: ', np.shape(target))
    # print('id2word: ', len(id2word))

    for i in range(target.size(0)):
        # print('batch:', i)
        # for j in range(len(target[i])):
        # each_target1 = []
        each_target = []
        for j in range(2): # 前10个是context，第11个是target
            # each_target = []
            # print('j:', j)
            add_sep = False
            each_utter_id = target[i][j]
            # print('each_utter_id:', each_utter_id)
            # print('each_utter_id_len:', len(each_utter_id))

            for each_id in each_utter_id[:20]:
                if each_id != PAD_ID and each_id != EOS_ID:
                    each_word = id2word[each_id]
                    # print(each_word)
                    # print(type(each_word))
                    if isinstance(each_word, str):
                    # each_target.append(each_word)
                        each_target.append(each_word.replace("'",""))
                        add_sep = True
                    else:
                        print(each_word)
                        print(type(each_word))
                        exit()
            if add_sep:
                each_target.append('<SEP>')
        each_target_1= " ".join(each_target)#.replace(' .','.')
        context_len = context_len + len(each_target_1.split(' '))  #context_len, num_context
        num_context = num_context + 1
        context_len_list.append(len(each_target_1.split(' ')))
        # each_target_1= each_target_1.replace(' ,',',') # ?
        # each_target_1= each_target_1.replace(' ?','?') # 
        # each_target_1= each_target_1.replace(' !','!') # 
        # print(len(each_target_1))
        batch_target.append(each_target_1)
        
        # # 64,10
        #     # print('each_target:', each_target)
        #     # print('each_target_len:', len(each_target))
        #     each_target_1= " ".join(each_target)
        #     # print('each_target_1:', each_target_1)
        #     each_target1.append(each_target_1)
        # # print('each_target1:', each_target1)
        # # print('each_target1_len:', len(each_target1))
        # # batch_target.append(each_target)
        # batch_target.append(each_target1)

        # exit()
    return batch_target, context_len, num_context, context_len_list

def context_know(context, knowledge, know_len, know_len_list):
    num_len = 0
    context_know_incor = []
    batch_size = len(context)
    for i in range(batch_size):
        each_context_know = []
        each_context = context[i]
        each_know = knowledge[i]
        # know_len
        each_len = len(each_know.split(' '))
        know_len = know_len + each_len
        know_len_list.append(each_len)

        each_context_know.append(each_context+each_know)
        each_context_know = " ".join(each_context_know).replace(' .','.')
        each_context_know= each_context_know.replace(' ,',',') # ?
        each_context_know= each_context_know.replace(' ?','?') # 
        each_context_know= each_context_know.replace(' !','!') #        
        context_know_incor.append(each_context_know)
        # print('each_context_know_len:', len(each_context_know.split(' ')))
        # num_len = num_len + len(each_context_know.split(' '))
    # print('total_num_len:', num_len)
    # print('num_len/batch_size:', num_len/batch_size)

    return context_know_incor, know_len, know_len_list


def text_train(
        # HyperConv_encoder, UtterFusion_encoder, domainHyper_encoder, KnowledgeIncor_encoder
        HyperConv_encoder: HyperConv,
        UtterFusion_encoder: UtterFusion,
        domainHyper_encoder: domainHyper,
        KnowledgeIncor_encoder: KnowledgeIncor,
        DynamicCorrelation_encoder: DynamicCorrelation,
        train_context_index,
        valid_context_index,
        test_context_index,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        model_file: str,
        vocab: Dict[str, int], 
        embed_init=None
):

    vocab_size = 4892  # num_node
    # 定义解码器的属性信息
    text_decoder_config = SimpleTextDecoderConfig(vocab_size, embed_init)
    # 讲输出映射到隐藏层
    to_hidden = ToHidden(text_decoder_config)
    to_hidden = to_hidden.to(GlobalConfig.device)
    # 定义具体的decoder，使用line60反馈回来的属性信息
    text_decoder = TextDecoder(text_decoder_config)
    text_decoder = text_decoder.to(GlobalConfig.device)

    # test pretrain BART
    # bleu_1_test = text_test(
    #         HyperConv_encoder,
    #         UtterFusion_encoder, 
    #         domainHyper_encoder, KnowledgeIncor_encoder, DynamicCorrelation_encoder,
    #         test_context_index,
    #         to_hidden,
    #         text_decoder,
    #         test_dataset,
    #         text_decoder_config.text_length,
    #         vocab)

    # exit()

    # path_domain_context = base_dir + 'Context/domain_context_' + 'train' + '.npy'
    # path_context_domain = base_dir + 'Context/context_domain_' + 'train' + '.npy'
    # path_global_repre_mode = path_global_repre + 'train' + '/'

    # domain_context = np.load(path_domain_context, allow_pickle=True).tolist()
    # context_domain = np.load(path_context_domain, allow_pickle=True).tolist()
    # # 因为一个dialogue可能会同时涉及多个domain，所以这里会有重复的可能性
    # domain_node_list1 = domain_node_shuffle(domain_context)
    # domain_node_list2 = [each_domain.replace('.npy','') for each_domain in domain_node_list1]
    # domain_node_dict = {}
    # # print('domain_node_list2-1:', len(domain_node_list2))
    # # 随机sample的每个domain的node
    # domain_global_repre_1 = []
    # domain_node_list2 = list(set(domain_node_list2))
    # # print('domain_node_list2-2:', len(domain_node_list2))

    # for domain_node in domain_node_list2:
    #     # dict_key = len(domain_node_dict) 
    #     # domain_node_dict[dict_key] = domain_node
    #     path_each_domain = path_global_repre_mode + domain_node + '.npy'
    #     each_global_repre = np.load(path_each_domain, allow_pickle=True).tolist()
    #     domain_node_dict[domain_node] = len(domain_node_dict) 
    #     # if isinstance(each_global_repre, list):
    #     #     domain_global_repre_1.append(each_global_repre)
    #     # else:
    #     #     each_global_repre = [float(0)]*2816
    #     domain_global_repre_1.append(each_global_repre)            
    # # print('domain_node_dict:', len(domain_node_dict))
    # subgraph_domain, domain_list =  domain_row_col(domain_node_dict, context_domain)  
    # # print('domain_global_repre_origin:', np.shape(domain_global_repre))
    # domain_global_repre = torch.stack([torch.tensor(each_repre, requires_grad=True) for each_repre in domain_global_repre_1])
    # for each_repre in domain_global_repre_1:
    #     print(np.shape(torch.tensor(each_repre, requires_grad=True)))
    
    # exit()

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TextTrainConfig.batch_size,
        shuffle=False,
        # num_workers=0)
        # shuffle=True, 
        num_workers=0)    

    # Model.
    vocab_size = 4892  # num_node
    # 定义解码器的属性信息
    text_decoder_config = SimpleTextDecoderConfig(vocab_size, embed_init)
    # 讲输出映射到隐藏层
    to_hidden = ToHidden(text_decoder_config)
    to_hidden = to_hidden.to(GlobalConfig.device)
    # 定义具体的decoder，使用line60反馈回来的属性信息
    text_decoder = TextDecoder(text_decoder_config)
    text_decoder = text_decoder.to(GlobalConfig.device)

    # Model parameters.
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        HyperConv_encoder, 
        UtterFusion_encoder, 
        DynamicCorrelation_encoder,
        domainHyper_encoder, 
        KnowledgeIncor_encoder,
        to_hidden,
        text_decoder
    ]]))

    optimizer = Adam(params, lr=TextTrainConfig.learning_rate)
    # optimizer = SGD(params, lr=TextTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None
    max_valid_bleu =None

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

    # Switch to train mode.----model.train()作用是启用batch normalization和drop out
    # ==注释
    # context_text_encoder.train()
    HyperConv_encoder.train()
    UtterFusion_encoder.train()
    DynamicCorrelation_encoder.train()
    domainHyper_encoder.train()
    KnowledgeIncor_encoder.train()
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

    output_file_train = open('text_127_train.out', 'w')
    output_file_train_name = 'text_127_train.out'


    finished = False
    # num_iterations = 10000
    for epoch_id in range(epoch_id, TextTrainConfig.num_iterations):
    # for epoch_id in range(epoch_id, 200):
        # print('num_iterations:', TextTrainConfig.num_iterations)
        # exit()
        context_len = 0
        num_context = 0
        context_len_list = []
        know_len = 0
        know_len_list = []
        num_1 = 0
        for batch_id, train_data in enumerate(train_data_loader):
            print('batch_id:', batch_id) 
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Set gradients to 0.--每个batch 参数初始化为0
            optimizer.zero_grad() #  # 将模型的参数梯度初始化为0
            texts, texts_vec1, text_lengths, batch_index, domain_label, text_venues = train_data
            context, context_len, num_context, context_len_list = target_id2word(texts, id2word, context_len, num_context, context_len_list)
            context2, know_len, know_len_list = context_know(context, text_venues[0], know_len, know_len_list)


            # print('texts:', texts)
            # print('text_venues:', text_venues)
            continue
            # print('context2:', context2)
            # print('texts_shape:', np.shape(texts))            
            # print('text_venues_shape:', np.shape(text_venues))            
            # exit()
            # torch.Size([64, 6])
            # print('domain_label: ', domain_label)
            # print('domain_label_shape: ', len(domain_label))
            # domain_label_shape:  torch.Size([64, 6])
            # print('domain_label_shape: ', np.shape(domain_label))
            # print('domain_label_type: ', type(domain_label))
            # exit()


            # num_1 = num_1 +1
            # if num_1 == 2:
            #     exit()
            # exit()
            # subgraphs_G = load_context_data(train_dataset.subgraphs, train_dataset.subgraphs_mapping_w, train_dataset.subgraphs_mapping_u, batch_index)
            # constsrcut domain subgraph
            texts_vec = texts_vec1[:,:-1,:]
            batch_size = np.shape(texts_vec)[0]
            # subgraphs_mapping_list = batch_index[0]
            # print('domain_global_repre-common:', np.shape(domain_global_repre))
            # print('subgraph_domain-common:', np.shape(subgraph_domain))
            # print('domain_list-common:', np.shape(domain_list))
            # print('domain_node_dict-common:', len(domain_node_dict))

            # domain_global_repre1, subgraphs_domain_node, subgraphs_domain_edge, domain_node_dict1 = construct_domain_hyper('train', subgraphs_mapping_list, domain_global_repre, subgraph_domain, domain_list, domain_node_dict)
            # domain_global_repre1: torch.Size([3132, 2816])
            # subgraphs_domain_node: (3132, 3132)
            # print('domain_global_repre1:', np.shape(domain_global_repre1))
            # print('subgraphs_domain_node:', np.shape(subgraphs_domain_node))
            # print('subgraphs_domain_edge:', np.shape(subgraphs_domain_edge))
            # print('domain_node_dict:', len(domain_node_dict))
            # exit()

            # To device.
            # 64,11,30
            # 64,11,25
            texts = texts.to(GlobalConfig.device)
            texts_vec = texts_vec.to(GlobalConfig.device)
            domain_label = domain_label.to(GlobalConfig.device)
            # 64,11
            # 是真实的文本长度
            text_lengths = text_lengths.to(GlobalConfig.device)
            # domain_global_repre2 = domain_global_repre1.float()
            # domain_global_repre2 = domain_global_repre2.to(GlobalConfig.device)

            context_len_train = 23369
    
            # context, context_know_repre, loss_domain = encode_context(
            #     HyperConv_encoder,
            #     UtterFusion_encoder, domainHyper_encoder, KnowledgeIncor_encoder, DynamicCorrelation_encoder,
            #     texts_vec,
            #     train_dataset, context_len_train, subgraphs_mapping_list, subgraphs_G, domain_global_repre2, subgraphs_domain_node, domain_node_dict1, domain_label
            # )
            texts1 = texts.transpose_(0, 1)[-1]
            # print('texts1:', np.shape(texts))
            # print('texts1:', texts1)
            # print('texts1_shape:', np.shape(texts1))
            text_lengths1 = text_lengths.transpose_(0, 1)[-1]
            # print('text_lengths1:', text_lengths1)
            # print('text_lengths1_shape:', np.shape(text_lengths1))

            # loss_gener = text_decoder(context, texts1, id2word, output_file_train)
            loss_gener = text_decoder(context2, texts1, id2word, output_file_train)

            # loss_gener, n_totals = text_loss(to_hidden, text_decoder, id2word,
            #                            text_decoder_config.text_length, context,
            #                            texts1, text_lengths1, context_know_repre, output_file_train, output_file_train_name, teacher_forcing_ratio=0)
            # command =  "sh eval_2.sh " + output_file_train_name
            #     # ref_file + "<" + cand_file + "> " + temp
            # try:
            #     # subprocess.call(command, shell=True) stdout=subprocess.PIPE).stdout
            #     # output = subprocess.run(command, shell=True).returncode
            #     # bleu_1_train = float((subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout).decode("gbk"))
            #     bleu_1_train1 = (subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout).decode("gbk")
            #     print('bleu_1_train1: ', bleu_1_train1)
            #     bleu_1_train = float((subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout).decode("gbk"))
            #     print('bleu_1_train: ', bleu_1_train)

            #     # print('bleu_1_train: ', bleu_1_train)
            #     # bleu_1 = bleu_1_1)
            #     # # output = subprocess.run(command, shell=True).args
            #     # print('bleu_1: ', bleu_1)
            #     # print(type(bleu_1))
            # except OSError as e:
            #     print("Execution failed:", e, file=sys.stderr)
            # exit()
            # print('bleu_1: %s, bleu_2: %s, bleu_3: %s, bleu_4: %s,' % (str(bleu_list[0]), str(bleu_list[1]), str(bleu_list[2]), str(bleu_list[3])))                    
            # loss = loss_gener + 100*loss_domain
            # sum_loss = sum_loss + loss_gener / text_decoder_config.text_length + 10*loss_domain
            sum_loss = loss_gener  
            # loss1.backward()
            # loss.backward()
            sum_loss.backward(retain_graph=True)
            optimizer.step()

            # Print loss every `TrainConfig.print_freq` batches.
            # if (batch_id + 1) % TextTrainConfig.print_freq == 0:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # sum_loss /= TextTrainConfig.print_freq
            # sum_loss = sum_loss/TextTrainConfig.print_freq
            print('epoch: {} \tbatch: {} \tloss: {} \tloss_gener: {}  \ttime: {}'.format(
                    epoch_id + 1, batch_id + 1, sum_loss, loss_gener,  cur_time))

            # print('epoch: {} \tbatch: {} \tloss: {} \tloss_gener: {} \tloss_domain: {} \ttime: {}'.format(
                    # epoch_id + 1, batch_id + 1, sum_loss, loss_gener/text_decoder_config.text_length, 10*loss_domain, cur_time))
            # print('bleu_1: %s, bleu_2: %s, bleu_3: %s, bleu_4: %s,' % (str(bleu_list[0]), str(bleu_list[1]), str(bleu_list[2]), str(bleu_list[3])))     
            # print('bleu_1_train: %s' % (str(bleu_1_train)))                    

            loss_dir = '/home/share/chenxiaolin/MultimodalDialogSystem/Model_data_dump10/loss_train_127.txt'

            f_loss = open(loss_dir, 'a')
            f_loss.write(str(sum_loss)+'\r\n')
            f_loss.close()                   
            # sum_loss = 0

            # print('domain_global_repre2_after_train:', domain_global_repre2[0])
            # print('domain_global_repre2_after_train:', np.shape(domain_global_repre2))
            # exit()

            # print('domain_global_repre_after_train:', domain_global_repre[0])
            # print('domain_global_repre_after_train:', np.shape(domain_global_repre))
            # exit()

            # Valid every `TrainConfig.valid_freq` batches.
            # 从原来设置的1000---100
            if (batch_id + 1) % TextTrainConfig.valid_freq == 0: #100
            # if (batch_id + 1) % 1 == 0:
                # 一个epoch验证一次
                bleu_1_valid = text_valid(
                                        HyperConv_encoder,
                                        UtterFusion_encoder, domainHyper_encoder, KnowledgeIncor_encoder, DynamicCorrelation_encoder, valid_context_index, vocab,
                                        to_hidden,
                                        text_decoder,
                                        valid_dataset,
                                        text_decoder_config.text_length)
                                    
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('time: {}'.format(cur_time))
                print('valid_bleu_1: %s ' % (str(bleu_1_valid)))                    
                # print('domain_global_repre_after_valid:', domain_global_repre[0])
                # print('domain_global_repre_after_valid:', np.shape(domain_global_repre))
                # exit()
                # Save current best model./当模型最好时进行test
                # if min_valid_loss is None or valid_loss < min_valid_loss:
                if max_valid_bleu is None or bleu_1_valid > max_valid_bleu:
                    # min_valid_loss = valid_loss
                    max_valid_bleu = bleu_1_valid
                    save_dict = {
                        'task': TEXT_TASK,
                        'epoch_id': epoch_id,
                        'min_valid_loss': min_valid_loss,
                        'max_valid_bleu': max_valid_bleu,
                        'optimizer': optimizer.state_dict(),
                        'HyperConv_encoder':
                            HyperConv_encoder.state_dict(),
                        'UtterFusion_encoder':
                            UtterFusion_encoder.state_dict(),
                        'DynamicCorrelation_encoder': DynamicCorrelation_encoder.state_dict(),
                        'domainHyper_encoder':
                            domainHyper_encoder.state_dict(),     
                        'KnowledgeIncor_encoder':
                            KnowledgeIncor_encoder.state_dict(),                                                    
                        'to_hidden':
                            to_hidden.state_dict(),
                        'text_decoder':
                            text_decoder.state_dict()
                    }
                    torch.save(save_dict, model_file)
                    print('Best model saved.')

                    bleu_1_test = text_test(
                            HyperConv_encoder,
                            UtterFusion_encoder, 
                            domainHyper_encoder, KnowledgeIncor_encoder, DynamicCorrelation_encoder,
                            test_context_index,
                            to_hidden,
                            text_decoder,
                            test_dataset,
                            text_decoder_config.text_length,
                            vocab)

                    # bleu_1_test = text_test(
                    #         HyperConv_encoder,
                    #         UtterFusion_encoder, 
                    #         domainHyper_encoder, KnowledgeIncor_encoder, DynamicCorrelation_encoder,
                    #         test_context_index,
                    #         to_hidden,
                    #         text_decoder,
                    #         test_dataset,
                    #         text_decoder_config.text_length,
                    #         vocab, domain_global_repre, subgraph_domain, domain_list, domain_node_dict)
                    print('\ttime: {}'.format(cur_time))
                    print('test_bleu_1: %s' % (str(bleu_1_test)))   

                    # print('test_bleu_1: %s, test_bleu_2: %s, test_bleu_3: %s, test_bleu_4: %s,' % (str(bleu_list[0]), str(bleu_list[1]), str(bleu_list[2]), str(bleu_list[3])))   
                    # print('domain_global_repre_after_test:', domain_global_repre[0])
                    # print('domain_global_repre_after_test:', np.shape(domain_global_repre))                    
                    # exit()                 
                else:
                    print('bad_loss_cnt:', bad_loss_cnt)
                    # bad_loss_cnt += 1
                    bad_loss_cnt = bad_loss_cnt + 1
                    # TextTrainConfig.patience
                    # if bad_loss_cnt > TextTrainConfig.patience:

                        # finished = True
                        # break

                # exit()
        # context_len
        path_write_len = 'context_len.txt'
        f_len = open(path_write_len, 'a')
        for each_len in context_len_list:
            f_len.write(str(each_len)+'\n')
        f_len.close() 
        # know_len
        path= 'context_know_len.txt'
        f_path = open(path, 'a')
        for each_know_len in know_len_list:
            f_path.write(str(each_know_len)+'\n')
        f_path.close()

        print('average_context_len:', context_len/num_context) # 25.57
        print('len_context_len_list:', len(context_len_list))
        print('know_len_aver: ', know_len/num_context) #75.48
        exit()
    output_file_train.close()