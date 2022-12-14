from typing import List

import numpy as np
import torch
from torch import nn


from torch.nn import functional as F



# class GlyphEmbedding(nn.Module):
#     """Glyph2id Embedding"""
#
#     def __init__(self, embedding_size: int, zixing_out_dim: int,zixing_id = get_zixing_ids()):
#         super(GlyphEmbedding, self).__init__()
#         zixing_ids = zixing_id
#         self.zixing_out_dim = zixing_out_dim
#         self.embedding = nn.Embedding(len(zixing_ids),embedding_size)
#         self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.zixing_out_dim, kernel_size=2,
#                               stride=1, padding=0)
#
#
#
#     def forward(self, zixing_ids):
#         """
#             get glyph images for batch inputs
#         Args:
#             input_ids: [batch, sentence_length]
#         Returns:
#             images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
#         """
#         embed = self.embedding(zixing_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
#         bs, sentence_length, zixing_locs, embed_size = embed.shape
#         view_embed = embed.view(-1, zixing_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
#         input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
#         # conv + max_pooling
#         zixing_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
#         zixing_embed = F.max_pool1d(zixing_conv, zixing_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
#         return torch.tensor(zixing_embed.view(bs, sentence_length, self.zixing_out_dim))


from typing import List

import numpy as np
import torch
from torch import nn
from getweibozixingid import get_all_word2id,produce_zixing_id_mask

from torch.nn import functional as F


class GlyphEmbedding(nn.Module):
    def __init__(self, embedding_size: int, zixing_out_dim: int, zixing_id = get_all_word2id()):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(GlyphEmbedding, self).__init__()
        self.zixing_ids = zixing_id
        self.zixing_out_dim = zixing_out_dim
        self.embedding = nn.Embedding((len(self.zixing_ids))+2, embedding_size)
        # self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.zixing_out_dim, kernel_size=2,
        #                       stride=1, padding=0)

#
    #
    def forward(self, zixing_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        embed = self.embedding(zixing_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length,zixing_locs,  embed_size = embed.shape
        mask0 = produce_zixing_id_mask()
        mask1 = torch.LongTensor(mask0)
        mask = torch.unsqueeze(mask1,3)
        embed = torch.matmul(embed, mask)
        zixing_embed = F.max_pool1d(embed,embed.shape[2])  # [(bs*sentence_length),pinyin_out_dim,1]
        return embed

        # view_embed = embed.view(bs,-1, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        # input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # # conv + max_pooling
        # zixing_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        # zixing_embed = F.max_pool1d(embed.shape[2])  # [(bs*sentence_length),pinyin_out_dim,1]
        #
        # return zixing_embed.view(bs, sentence_length, embed_size)

# import numpy as np
# zixing_id = get_all_word2id()
# print(zixing_id)
# print(np.array(zixing_id).shape)
# print(np.array(zixing_id).size)
# glyph_embeddings = GlyphEmbedding(embedding_size=768,zixing_out_dim=768)
# embedding = glyph_embeddings.embedding(torch.LongTensor(zixing_id))
# print(embedding)
# print(embedding.shape)
# print(glyph_embeddings.weight)