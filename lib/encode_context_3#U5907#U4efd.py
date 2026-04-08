import torch
import numpy as np
from config import DatasetConfig, GlobalConfig
from constant import SOS_ID
from model import TextEncoder, ImageEncoder, ContextEncoder, HyperConv, UtterFusion, domainHyper, KnowledgeIncor


def encode_context(context_text_encoder: TextEncoder,
                   HyperConv_encoder: HyperConv,
                   UtterFusion_encoder: UtterFusion, domainHyper_encoder: domainHyper, KnowledgeIncor_encoder,
                   texts_vec, train_dataset, context_len, subgraphs_mapping_list, subgraphs_G, train_global, subgraphs_domain_node, domain_node_dict):
    """ Encode context.
    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Context encoder.
        texts: Texts (dialog_context_size + 1, batch_size, dialog_text_max_len)
        text_lengths: Text lengths (dialog_context_size + 1, batch_size)
        images: Images (dialog_context_size + 1, pos_images_max_num,
                        batch_size, 3, image_size, image_size)

    Returns:
        Context vector (batch_size, context_vector_size)
          context_vector_size = hidden_size * num_layers * num_directions

    """
    # print('-------HyperConv_encoder begin----------------')
    # 对于batch中的每一个context分别计算对应的embedding
    edge_embeddings = HyperConv_encoder(subgraphs_mapping_list, subgraphs_G, train_dataset.reversed_subgraphs_mapping_w, context_len)

    edge_embeddings = edge_embeddings.to(GlobalConfig.device)

    texts_vec = texts_vec.float()

    # utter_fusion: torch.Size([16, 300])
    utter_fusion = UtterFusion_encoder(texts_vec, edge_embeddings)
    # utter_fusion = torch.randn(16, 300)
    # print('utter_fusion:', np.shape(utter_fusion))

    # context: torch.Size([16, 300])
    context = domainHyper_encoder(train_global, utter_fusion, subgraphs_mapping_list, subgraphs_domain_node, domain_node_dict)
    # print('context:', np.shape(context))

    # ====
    # print('utter_fusion:', utter_fusion[0][:])
    # print('context:', np.shape(context))    
    # 64,300
    # hiddens = context
    # context-based knowledge representation
    # context_know_repre = context

    context_know_repre = KnowledgeIncor_encoder(context)
    # context_know_repre = context
    # print('utter_fusion: ', np.shape(utter_fusion))
    # print('local_texts_vec: ', np.shape(texts_vec))
    # print('batch_len_word: ', len(batch_len_word))
    # print('batch_len_word: ', len(batch_len_word))
    # exit()

    # ===以下内容
    # batch_size = texts.size(1)

    # sos = SOS_ID * torch.ones(batch_size, dtype=torch.long).view(-1, 1)
    # sos = sos.to(GlobalConfig.device)
    # # (batch_size, 1)
    # ===以上内容
    # context = []
    # hiddens = None
    # # 这里是对一个context中的每一条utterance进行遍历----那怎么处理不同context的内容呢---噢噢 知道了，前边进行了转置，所以用index可以取出所有context第i条utterance
    # for i in range(DatasetConfig.dialog_context_size):
    #     text, text_length = texts[i], text_lengths[i]
    #     text = text.to(GlobalConfig.device)
    #     text_length = text_length.to(GlobalConfig.device)
    #     # text: (batch_size, dialog_text_max_len)
    #     # text_length: (batch_size, )

    #     # Insert SOS (Start Of Sentence) to the start of sentence
    #     text = torch.cat((sos, text), 1).to(GlobalConfig.device)
    #     text_length += 1
    #     # 其实这个地方才是真正的调用编码方法(forward)
    #     # 通过当前方法控制输入--控制输进model的具体text
    #     encoded_text, hiddens = context_text_encoder(text, text_length)
    #     encoded_text = encoded_text.to(GlobalConfig.device)
    #     # (batch_size, text_feat_size)
    #     # text_feat_size = hidden_size * num_layers * num_directions

    #     # 对每张图片，都是用文本为其分配权重
    #     for j in range(DatasetConfig.pos_images_max_num):
    #         image = images[i][j]
    #         # (batch_size, 3, image_size, image_size)

    #         encoded_image = context_image_encoder(image, encoded_text)
    #         encoded_image = encoded_image.to(GlobalConfig.device)
    #         # (batch_size, )
    #         # torch.cat是拼接操作，concatenate的缩写
    #         mm = torch.cat((encoded_text, encoded_image), 1)
    #         mm = mm.to(GlobalConfig.device)
    #         # (batch_size, text_feat_size + image_feat_size)

    #         context.append(mm)

    # context = torch.stack(context)
    # context = context.to(GlobalConfig.device)
    # # (dialog_context_size, batch_size, text_feat_size + image_feat_size)

    # context = context_encoder(context)
    # ===以上内容
    # (batch_size, context_vector_size)
    # context_vector_size = hidden_size * num_layers * num_directions

    return context, context_know_repre
    # return utter_fusion, utter_fusion

