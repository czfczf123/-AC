# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)
import torch
import torch.nn as nn

from repcodec.repcodec.modules.decoder import Decoder
from repcodec.repcodec.modules.encoder import Encoder
from repcodec.repcodec.modules.projector import Projector
from repcodec.repcodec.modules.quantizer import Quantizer,Sub_Quantizer


class RepCodec(nn.Module):
    def __init__(
            self,
            input_channels=768,
            output_channels=768,
            encode_channels=768,
            decode_channels=768,
            code_dim=768,
            codebook_num=1,
            codebook_size=512,
            bias=True,
            enc_ratios=(1, 1),
            dec_ratios=(1, 1),
            enc_strides=(1, 1),
            dec_strides=(1, 1),
            enc_kernel_size=3,
            dec_kernel_size=3,
            enc_block_dilations=(1, 1),
            enc_block_kernel_size=3,
            dec_block_dilations=(1, 1),
            dec_block_kernel_size=3,
            num_quantizers_sub=None
    ):
        super().__init__()

        self.input_channels = input_channels

        self.encoder = Encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            channel_ratios=enc_ratios,
            strides=enc_strides,
            kernel_size=enc_kernel_size,
            bias=bias,
            block_dilations=enc_block_dilations,
            unit_kernel_size=enc_block_kernel_size
        )

        self.decoder = Decoder(
            code_dim=code_dim,
            output_channels=output_channels,
            decode_channels=decode_channels,
            channel_ratios=dec_ratios,
            strides=dec_strides,
            kernel_size=dec_kernel_size,
            bias=bias,
            block_dilations=dec_block_dilations,
            unit_kernel_size=dec_block_kernel_size
        )

        self.projector = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False
        )

        self.quantizer = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size
        )
        '''

        self.quantizer_sub=nn.ModuleList([Sub_Quantizer(code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=64) for _ in range(num_quantizers_sub)])

        '''

    def forward_gai(self, x,lid):

        x = self.encoder(x)
        z = self.projector(x)

        #vq_loss_all=torch.zeros(x.shape[0],device=x.device)
        zq_all=torch.zeros_like(x)
        indices=torch.zeros(x.shape[0],1,x.shape[2],device=x.device)


        for i in range(len(lid)):

            zq, indice=self.quantizer_sub[lid[i]].inference(z[i:i+1],self.quantizer.codebook.get_codebook(),lid[i])
            indices[i]=indice
            zq_all[i]=zq
            #print('IND',indice.shape)
        loss_vq=nn.functional.mse_loss(z.detach(),zq_all.detach())

        return indices,loss_vq

    def forward(self, x):
        #x = self.encoder(x)
        #z = self.projector(x)
        zq, vqloss, perplexity = self.quantizer(x)
        y = self.decoder(zq)
        return y, zq, vqloss, perplexity
    def encode_id(self, x):
        #x = self.encoder(x)
        #z = self.projector(x)
        zq, indices = self.quantizer.inference(x)
        return zq, indices

    def decode(self, indices):
        indices = indices.unsqueeze(0)
        z = self.quantizer.decode(indices)
        # print("rep_z",z.shape)
        z = z.squeeze(0)
        y = self.decoder(z.transpose(1, -1))

        return z,y
    def get_codebook(self):
        return self.quantizer.codebook.get_codebook()
