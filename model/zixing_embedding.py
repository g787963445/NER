from typing import List

import numpy as np
import torch
from torch import nn
from getweibozixingid import get_all_word2id,produce_zixing_id_mask,zixingtoid

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
#         return zixing_embed.view(bs, sentence_length, self.zixing_out_dim)

zixing2id,id2zixing = zixingtoid()
class GlyphEmbedding(nn.Module):
    def __init__(self, embedding_size: int, zixing_out_dim: int, zixing_id =zixing2id):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(GlyphEmbedding, self).__init__()
        self.zixing_id = zixing_id
        self.zixing_out_dim = zixing_out_dim
        self.embedding = nn.Embedding((len(self.zixing_id))+2, embedding_size)
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

        embed = self.embedding(zixing_ids)  # [bs,sentence_length,16,embed_size]
        device = embed.device if embed is not None else embed.device
        bs, sentence_length, zixing_locs, embed_size = embed.shape
        mask0 = produce_zixing_id_mask(zixing_ids)
        mask1 = torch.tensor(mask0)
        mask = torch.unsqueeze(mask1, 2)
        mask = torch.tensor(mask)  # 463,256,1,16

        view_embed = embed.view(-1, zixing_locs, embed_size)  # [(bs*sentence_length),16,embed_size]
        mask = mask.view(-1, zixing_locs, 1)
        embed = torch.mul(view_embed, mask) #bs*sentence_length,16,emb
        input_embed = embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, 16]
        zixing_embed = F.max_pool1d(input_embed, input_embed.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        zixing = zixing_embed.view(bs, sentence_length, 768)

        return zixing
