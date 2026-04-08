import torch
import numpy as np
from config import DatasetConfig, GlobalConfig
from constant import SOS_ID
from model import TextEncoder, ImageEncoder, ContextEncoder, HyperConv, UtterFusion, domainHyper, KnowledgeIncor, DynamicCorrelation

# def encode_context(
#                    HyperConv_encoder: HyperConv,
#                    UtterFusion_encoder: UtterFusion, domainHyper_encoder: domainHyper, KnowledgeIncor_encoder,
#                    texts_vec, train_dataset, context_len, subgraphs_mapping_list, subgraphs_G, train_global, subgraphs_domain_node, domain_node_dict):
#     """ Encode context.
#     Args:
#         # context_text_encoder (TextEncoder): Context text encoder.
#         context_image_encoder (ImageEncoder): Context image encoder.
#         context_encoder (ContextEncoder): Context encoder.
#         texts: Texts (dialog_context_size + 1, batch_size, dialog_text_max_len)
#         text_lengths: Text lengths (dialog_context_size + 1, batch_size)
#         images: Images (dialog_context_size + 1, pos_images_max_num,
#                         batch_size, 3, image_size, image_size)

#     Returns:
#         Context vector (batch_size, context_vector_size)
#           context_vector_size = hidden_size * num_layers * num_directions

#     """
#     # print('-------HyperConv_encoder begin----------------')
#     # 对于batch中的每一个context分别计算对应的embedding
#     edge_embeddings = HyperConv_encoder(texts_vec, subgraphs_mapping_list, subgraphs_G, train_dataset.reversed_subgraphs_mapping_w, context_len)
#     edge_embeddings = edge_embeddings.to(GlobalConfig.device)
#     # 32, 10, 300

#     edge_embeddings1 = edge_embeddings.transpose(0, 1)
#     # print('edge_embeddings1:', np.shape(edge_embeddings1))

#     # utter_fusion: torch.Size([16, 300])
#     # 经过BiLSTM编码之后的dialogue表征
#     utter_fusion = UtterFusion_encoder(edge_embeddings1)
#     # context = domainHyper_encoder(train_global, utter_fusion, subgraphs_mapping_list, subgraphs_domain_node, domain_node_dict)
#     # context_know_repre, score_MIM = KnowledgeIncor_encoder(context, venue_related_repre, value_list_extend2)
#     context = utter_fusion
#     context_know_repre = utter_fusion
#     score_MIM = torch.tensor(0)
#     return context, context_know_repre, score_MIM
#     # return utter_fusion, utter_fusion

# ===origin===
def encode_context(
                   HyperConv_encoder: HyperConv,
                   UtterFusion_encoder: UtterFusion, domainHyper_encoder: domainHyper, KnowledgeIncor_encoder, DynamicCorrelation_encoder:DynamicCorrelation,
                   texts_vec, train_dataset, context_len, subgraphs_mapping_list, subgraphs_G, train_global, subgraphs_domain_node, domain_node_dict, domain_label):
    """ Encode context.
    Args:
        # context_text_encoder (TextEncoder): Context text encoder.
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
    # ==移除utterance edge_embeddings = HyperConv_encoder(texts_vec, subgraphs_mapping_list, subgraphs_G, train_dataset.reversed_subgraphs_mapping_w, context_len)
    edge_embeddings, venue_embeddings = HyperConv_encoder(texts_vec, subgraphs_mapping_list, subgraphs_G, train_dataset.reversed_subgraphs_mapping_w, context_len)
    # edge_embeddings: torch.Size([32, 10, 300])
    edge_embeddings = edge_embeddings.to(GlobalConfig.device)
    # 32, 10, 300
    # print('edge_embeddings:', np.shape(edge_embeddings))
    # exit()
    # texts_vec = texts_vec.float()
    # print('edge_embeddings:', np.shape(edge_embeddings))
    # print('venue_embeddings:', np.shape(venue_embeddings))
    # print('venue_related_repre:', np.shape(venue_related_repre))
    # print('value_list_extend2:', np.shape(value_list_extend2))

    edge_embeddings1 = edge_embeddings.transpose(0, 1)
    # print('edge_embeddings1:', np.shape(edge_embeddings1))

    # utter_fusion: torch.Size([16, 300])
    # 经过BiLSTM编码之后的dialogue表征
    # [64, 6, 300]
    utter_fusion = UtterFusion_encoder(edge_embeddings1)
    # print(utter_fusion)
    # print(np.shape(utter_fusion))
    # exit()
    # context = utter_fusion
    # utter_fusion: torch.Size([32, 300])
    # print('utter_fusion:', np.shape(utter_fusion))
    # context: torch.Size([16, 300])
    # context = utter_fusion
    final_repre_update, loss_return = DynamicCorrelation_encoder(utter_fusion, domain_label)
    context = final_repre_update
    context_know_repre = final_repre_update
    return context, context_know_repre, loss_return
    # exit()
    #==注释掉之前的 context = domainHyper_encoder(train_global, utter_fusion, subgraphs_mapping_list, subgraphs_domain_node, domain_node_dict)
    # print('context:', np.shape(context))
    # exit()
    # ====
    # print('utter_fusion:', utter_fusion[0][:])
    # print('context:', np.shape(context))    
    # 64,300
    # hiddens = context
    # context-based knowledge representation
    # context_know_repre = context
    # 此处的know_repre是context和knowledge级联之后的表征
    # context_know_repre, score_MIM = KnowledgeIncor_encoder(context, venue_related_repre, value_list_extend2)
    # context_know_repre, score_MIM = KnowledgeIncor_encoder(context)

    score_MIM = 0
    context_know_repre = context
    # print('context_know_repre:', np.shape(context_know_repre))


    return context, context_know_repre, score_MIM
    # return utter_fusion, utter_fusion

