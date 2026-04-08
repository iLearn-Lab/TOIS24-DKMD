from datetime import datetime
from itertools import chain
from os.path import isfile
from typing import List, Dict
# import datetime
import torch
from torch.optim import Adam
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
from model import TextEncoder, ImageEncoder, ContextEncoder, HyperConv, UtterFusion, domainHyper, KnowledgeIncor
import numpy as np
import random
import scipy.sparse as sp


base_dir = '/home/share/chenxiaolin/MultimodalDialogSystem/Model/dataset_dump/'
path_global_repre = base_dir + 'Context/global_repre/'


def get_context_dialog(context):
    dialog = context.get('dialogues')
    text_list: List[List[int]] = [utter.text for utter in dialog]
    text_length_list: List[int] = [utter.text_len for utter in dialog]
    text_vec_list = [np.squeeze(utter.text_vec).tolist() for utter in dialog]
    texts = torch.stack(tuple([torch.tensor(text) for text in text_list]))
    texts_vec =  torch.stack([torch.LongTensor(text_vec) for text_vec in text_vec_list])
    text_lengths = torch.tensor(text_length_list)
    utter_type = dialog[-2].utter_type
    return texts, texts_vec, text_lengths, utter_type

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
    for context_index in index_list[0]:
        context_index = int(context_index)
        col = subgraphs[context_index]['u']  # u是超边--utterance
        row = subgraphs[context_index]['w']  # i是结点--代表item---word/image
        data = np.ones(len(col))
        # print('len(subgraphs_mapping_w[context_index]', len(subgraphs_mapping_w[context_index]))
        sg = sp.coo_matrix((data, (row, col)), shape=(len(subgraphs_mapping_w[context_index]), len(subgraphs_mapping_u[context_index])))
        # print('Done constructing subgraph', str(context_index))
        # print(len(subgraphs_mapping_w[context_index]), len(subgraphs_mapping_u[context_index]), len(data))
        subgraphs_G[context_index] = {}
        subgraphs_G[context_index]['G'], subgraphs_G[context_index]['E'] = generate_G_from_H(sg)
    return subgraphs_G

def domain_node_shuffle(domain_context):
    batch_list = []
    domain_node_0 = random.sample(domain_context[0], 1539)
    domain_node_1 = random.sample(domain_context[1], 340) 
    domain_node_2 = random.sample(domain_context[2], 253) 
    domain_node_3 = random.sample(domain_context[3], 324) 
    domain_node_4 = random.sample(domain_context[4], 726)
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
    print('domain_list_len:', len(domain_list))
    print('subgraph_domain:', len(subgraph_domain))
    return subgraph_domain, domain_list


def generate_G_from_H(H):
    # H的维度是节点*边--N*M
    # 边的数目
    n_edge = H.shape[1]
    # the weight of the hyperedge
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
    # --对角线元素为这个，其余的为0---M*M
    invDE2 = sp.diags(np.power(DE, -0.5).flatten())
    # N*N
    DV2 =  sp.diags(np.power(DV, -0.5).flatten())
    # M*M
    W = sp.diags(W)
    # M*N
    HT = H.T

    # DV2是干啥的---感觉这个的G是指的做完部分超图卷积之后的矩阵(相对于完整的超图卷积，还没有乘表示)
    invDE_HT_DV2 = invDE2 * HT * DV2
    G = DV2 * H * W * invDE2 * invDE_HT_DV2
    return G, invDE_HT_DV2

def construct_domain_hyper(mode, subgraphs_mapping_list):
    # subgraphs_mapping_list = list(subgraphs_mapping_list)
    path_domain_context = base_dir + 'Context/domain_context_' + mode + '.npy'
    path_context_domain = base_dir + 'Context/context_domain_' + mode + '.npy'
    path_global_repre_mode = path_global_repre + mode + '/'

    domain_context = np.load(path_domain_context, allow_pickle=True).tolist()
    context_domain = np.load(path_context_domain, allow_pickle=True).tolist()
    # print('domain_context:', (domain_context))
    # print('subgraphs_mapping_list:', subgraphs_mapping_list)
    # print('domain_context:', len(domain_context))
    # print('context_domain:', len(context_domain))
    # print('subgraphs_mapping_list:', len(subgraphs_mapping_list))

    domain_node_list1 = domain_node_shuffle(domain_context)
    domain_node_list1 = [each_domain.replace('.npy','') for each_domain in domain_node_list1]
    # print('domain_node_list1:', len(domain_node_list1))
    # print('domain_node_list1:', type(domain_node_list1))
    # print('subgraphs_mapping_list_len:', len(list(subgraphs_mapping_list)))
    # print('subgraphs_mapping_list_type:', type(list(subgraphs_mapping_list)))
    # print('subgraphs_mapping_list_set:', len(set(list(subgraphs_mapping_list))))
    # print('subgraphs_mapping_list_set_len:', len(list(set(list(subgraphs_mapping_list)))))

    domain_node_list1.extend(list(subgraphs_mapping_list))
    # print('domain_node_list:', domain_node_list1)
    domain_node_list = list(set(domain_node_list1))
    # domain_node_list = domain_node_list1
    
    # print('domain_node_list:', len(domain_node_list))

    domain_node_dict = {}
    domain_global_repre = []
    for domain_node in domain_node_list:
        # dict_key = len(domain_node_dict) 
        # domain_node_dict[dict_key] = domain_node
        domain_node_dict[domain_node] = len(domain_node_dict) 
        path_each_domain = path_global_repre_mode + domain_node + '.npy'
        each_global_repre = np.load(path_each_domain, allow_pickle=True).tolist()
        domain_global_repre.append(each_global_repre)
    # print('domain_node_dict:', len(domain_node_dict))
    # print('domain_global_repre:', np.shape(domain_global_repre))

    subgraph_domain, domain_list =  domain_row_col(domain_node_dict, context_domain)  

    subgraphs_G_domain = {}
    row_domain = domain_list   
    col_domain = subgraph_domain
    # col_domain = domain_list
    # row_domain = subgraph_domain
    data_domain = np.ones(len(col_domain))
    # print('data_domain:', np.shape(data_domain))

    # sg_domain = sp.coo_matrix((data_domain, (row_domain, col_domain)), shape=(6, len(train_dialogs)))
    # sg_domain = sp.coo_matrix((data_domain, (row_domain, col_domain)), shape=(5, len(train_dialogs)))
    sg_domain = sp.coo_matrix((data_domain, (row_domain, col_domain)), shape=(len(domain_node_list) , 5))
    subgraphs_domain_node, subgraphs_domain_edge = generate_G_from_H(sg_domain)
    # print('subgraphs_domain_node:', np.shape(subgraphs_domain_node))
    # print('subgraphs_domain_edge:', np.shape(subgraphs_domain_edge))

    # repre
    domain_global_repre_list =  torch.stack([torch.LongTensor(each_repre) for each_repre in domain_global_repre])
    # print('domain_global_repre_list:', np.shape(domain_global_repre_list))

    return domain_global_repre_list, subgraphs_domain_node, subgraphs_domain_edge, domain_node_dict

def text_train(
        context_text_encoder: TextEncoder, # context_text_encoder, HyperConv_encoder, UtterFusion_encoder, domainHyper_encoder, KnowledgeIncor_encoder
        HyperConv_encoder: HyperConv,
        UtterFusion_encoder: UtterFusion,
        domainHyper_encoder: domainHyper,
        KnowledgeIncor_encoder: KnowledgeIncor,
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
    """Text train.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        train_dataset (Dataset): Train dataset.
        valid_dataset (Dataset): Valid dataset.
        test_dataset (Dataset): Test dataset.
        model_file (str): Saved model file.
        vocab (Dict[str, int]): Vocabulary.
        embed_init: Initial embedding (vocab_size, embed_size).

    """
    # HyperConv_encoder()
    # Data loader.
    # train_config文件中--batch:64;num_data_loader_workers:5
    # DataLoader本质上就是一个iterable（跟python的内置类型list等一样），并利用多进程来加速batch data的处理，使用yield来使用有限的内
    # train_data_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=TextTrainConfig.batch_size,
    #     shuffle=True,
    #     num_workers=TextTrainConfig.num_data_loader_workers) 
    # train_global =  torch.stack([torch.LongTensor(each_repre) for each_repre in train_global_repre])
    # valid_global =  torch.stack([torch.LongTensor(each_repre) for each_repre in valid_global_repre])
    # test_global =  torch.stack([torch.LongTensor(each_repre) for each_repre in test_global_repre])
    # tip_vec_repre =  torch.stack([torch.LongTensor(each_repre) for each_repre in tip_vec]) #tip_vec_repre, triplet_vec_repre
    # triplet_vec_repre =  torch.stack([torch.LongTensor(each_repre) for each_repre in triplet_vec])
    

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TextTrainConfig.batch_size,
        shuffle=False,
        # shuffle=True, 
        num_workers=TextTrainConfig.num_data_loader_workers)
        # num_workers=1)    

    # Model.
    vocab_size = len(vocab)  # num_node
    # 定义解码器的属性信息
    text_decoder_config = SimpleTextDecoderConfig(vocab_size, embed_init)
    to_hidden = ToHidden(text_decoder_config)
    to_hidden = to_hidden.to(GlobalConfig.device)
    # 定义具体的decoder，使用line60反馈回来的属性信息
    text_decoder = TextDecoder(text_decoder_config)
    text_decoder = text_decoder.to(GlobalConfig.device)

    # Model parameters.
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        context_text_encoder, 
        HyperConv_encoder, 
        UtterFusion_encoder, 
        domainHyper_encoder, 
        KnowledgeIncor_encoder,
        to_hidden,
        text_decoder
    ]]))

    optimizer = Adam(params, lr=TextTrainConfig.learning_rate)
    epoch_id = 0
    min_valid_loss = None

    # Load saved state.
    if isfile(model_file):
        state = torch.load(model_file)
        to_hidden.load_state_dict(state['to_hidden'])
        text_decoder.load_state_dict(state['text_decoder'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_id = state['epoch_id']
        min_valid_loss = state['min_valid_loss']

    # Loss.
    sum_loss = 0
    bad_loss_cnt = 0
    batch_size = 32

    # Switch to train mode.----model.train()作用是启用batch normalization和drop out
    context_text_encoder.train()
    HyperConv_encoder.train()
    UtterFusion_encoder.train()
    domainHyper_encoder.train()
    KnowledgeIncor_encoder.train()
    to_hidden.train()
    text_decoder.train()

    finished = False
    # num_iterations = 10000
    for epoch_id in range(epoch_id, TextTrainConfig.num_iterations):
    # for epoch_id in range(epoch_id, 200):
        # print('num_iterations:', TextTrainConfig.num_iterations)
        # exit()
        for batch_id, train_data in enumerate(train_data_loader): 
        # for batch_id in range(0, len(train_context_index), TextTrainConfig.batch_size):
        # for batch_id in range(0, len(train_context_index), batch_size):   
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # print('1111111111')
            # Set gradients to 0.--每个batch 参数初始化为0
            optimizer.zero_grad()
            # print('22222222')
            # index_list = train_context_index[batch_id, batch_id+batch_size]
            # print('index_list:', index_list)
            # load-数据--context|subgraph_construct;
            # load_context_data('train', index_list, train_dataset.subgraphs, train_dataset.subgraphs_mapping_w, train_dataset.subgraphs_mapping_u)
            # exit()
            # print('batch_id: ', batch_id)
            # print(np.shape(train_data))
            # # print('train_data:', type(train_data))
            # for i in train_data:
            #     # print('----'+str(i)+'----')
            #     print(i)
            # exit()
            # texts, texts_vec, text_lengths, images, utter_types = train_data
            texts, texts_vec, text_lengths, batch_index = train_data
            # print(type(train_dataset.subgraphs))
            # print('train_dataset.subgraphs.keys: ', list(train_dataset.subgraphs.keys())[0])
            # print('train_dataset.subgraphs.keys: ', type(list(train_dataset.subgraphs.keys())[0]))
            # print('train_dataset.subgraphs.get(0).keys--int: ', train_dataset.subgraphs.get(9257).keys())
            # print('train_dataset.subgraphs,get(0).keys--str: ', train_dataset.subgraphs.get('9257').keys())
            subgraphs_G = load_context_data(train_dataset.subgraphs, train_dataset.subgraphs_mapping_w, train_dataset.subgraphs_mapping_u, batch_index)
            # constsrcut domain subgraph
            
            # print('texts:', np.shape(texts))
            # print('texts_vec:', np.shape(texts_vec))
            # print('text_lengths:', np.shape(text_lengths))
            # print('batch_index:', np.shape(batch_index))
            # print('subgraphs_G:', np.shape(subgraphs_G))
            # print('subgraphs_G:', type(subgraphs_G))
            # exit()
            texts_vec = texts_vec[:,:-1,:]
            # print('texts_vec:', np.shape(texts_vec))
            batch_size = np.shape(texts_vec)[0]
            # print('batch_size:', batch_size)
            # texts, texts_vec, text_lengths, images, utter_types = train_data
            # subgraphs_mapping_list = extract_graph_index(batch_id, batch_size)
            subgraphs_mapping_list = batch_index[0]
            domain_global_repre, subgraphs_domain_node, subgraphs_domain_edge, domain_node_dict = construct_domain_hyper('train', subgraphs_mapping_list)
            # print('-----subgraphs_mapping_list-------')
            # print('domain_node_dict:', domain_node_dict)
            # print('domain_global_repre:', np.shape(domain_global_repre))
            # print('subgraphs_domain_node:', np.shape(subgraphs_domain_node))
            # print('subgraphs_domain_edge:', np.shape(subgraphs_domain_edge))
            # print('domain_node_dict:', np.shape(domain_node_dict))
            # exit()
            # To device.
            # 64,11,30
            texts = texts.to(GlobalConfig.device)
            texts_vec = texts_vec.to(GlobalConfig.device)
            # 64,11
            text_lengths = text_lengths.to(GlobalConfig.device)
            domain_global_repre = domain_global_repre.to(GlobalConfig.device)

            # subgraphs_G = subgraphs_G.to(GlobalConfig.device)
            # images = images.to(GlobalConfig.device)

            # utter_types = utter_types.to(GlobalConfig.device)

            # texts.transpose_(0, 1)
            # # (dialog_context_size + 1, batch_size, dialog_text_max_len)

            # text_lengths.transpose_(0, 1)
            # # (dialog_context_size + 1, batch_size)
            context_len_train = 23369
            # images.transpose_(0, 1)
            # images.transpose_(1, 2)
            # (dialog_context_size + 1, pos_images_max_num, batch_size, 3,
            #  image_size, image_size)
            # print('-----encode_context-------')
            # Encode context.---这个地方才是把之前定义好的各种模型放到一起
            # context: torch.Size([64, 2048])
            # hiddens: torch.Size([31, 64, 512])
            # context_know_repre
            context, context_know_repre = encode_context(
                context_text_encoder,
                HyperConv_encoder,
                UtterFusion_encoder, domainHyper_encoder, KnowledgeIncor_encoder,
                texts_vec,
                train_dataset, context_len_train, subgraphs_mapping_list, subgraphs_G
            )
            # (batch_size, context_vector_size)
            # 最后一条是text----所以构造的应该最后一个是目标
            texts.transpose_(0, 1)
            text_lengths.transpose_(0, 1)
            loss, n_totals = text_loss(to_hidden, text_decoder,
                                       text_decoder_config.text_length, context,
                                       texts[-1], text_lengths[-1], context_know_repre)
            sum_loss += loss / text_decoder_config.text_length

            loss.backward()
            optimizer.step()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % TextTrainConfig.print_freq == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sum_loss /= TextTrainConfig.print_freq
                print('epoch: {} \tbatch: {} \tloss: {} \ttime: {}'.format(
                    epoch_id + 1, batch_id + 1, sum_loss, cur_time))
                sum_loss = 0

            # Valid every `TrainConfig.valid_freq` batches.
            # 从原来设置的1000---100
            if (batch_id + 1) % TextTrainConfig.valid_freq == 0:
                valid_loss = text_valid(context_text_encoder,
                                        HyperConv_encoder,
                                        UtterFusion_encoder, domainHyper_encoder, KnowledgeIncor_encoder, valid_context_index,
                                        to_hidden,
                                        text_decoder,
                                        valid_dataset,
                                        text_decoder_config.text_length)
            # context, hiddens = encode_context(
            #     context_text_encoder,
            #     HyperConv_encoder,
            #     UtterFusion_encoder, domainHyper_encoder,
            #     context_image_encoder,
            #     context_encoder,
            #     texts_vec,
            #     images,
            #     subgraphs_mapping_list, train_global
            # )                                        
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print('valid_loss: {} \ttime: {}'.format(valid_loss, cur_time))

                # Save current best model.
                if min_valid_loss is None or valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    # bad_loss_cnt = 0
                    save_dict = {
                        'task': TEXT_TASK,
                        'epoch_id': epoch_id,
                        'min_valid_loss': min_valid_loss,
                        'optimizer': optimizer.state_dict(),

                        'context_text_encoder':
                            context_text_encoder.state_dict(),
                        'context_image_encoder':
                            context_image_encoder.state_dict(),
                        'context_encoder':
                            context_encoder.state_dict(),
                        'to_hidden':
                            to_hidden.state_dict(),
                        'text_decoder':
                            text_decoder.state_dict()
                    }
                    torch.save(save_dict, model_file)
                    print('Best model saved.')
                else:
                    print('bad_loss_cnt:', bad_loss_cnt)
                    bad_loss_cnt += 1
                    # TextTrainConfig.patience
                    if bad_loss_cnt > TextTrainConfig.patience:
                        text_test(context_text_encoder,
                                  HyperConv_encoder,
                                  UtterFusion_encoder, 
                                  domainHyper_encoder, KnowledgeIncor_encoder, 
                                  test_context_index,
                                  to_hidden,
                                  text_decoder,
                                  test_dataset,
                                  text_decoder_config.text_length,
                                  vocab)
                        finished = True
                        break
        if finished:
            break
