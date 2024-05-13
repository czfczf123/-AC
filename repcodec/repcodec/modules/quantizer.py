# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)
import random

import torch
import torch.nn as nn

from repcodec.repcodec.layers.vq_module import ResidualVQ


class Quantizer(nn.Module):
    def __init__(
            self,
            code_dim: int,
            codebook_num: int,
            codebook_size: int,
    ):
        super().__init__()
        self.codebook = ResidualVQ(
            dim=code_dim,
            num_quantizers=codebook_num,
            codebook_size=codebook_size
        )

    def initial(self):
        self.codebook.initial()

    def forward(self, z):
        zq, vqloss, perplexity = self.codebook(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, vqloss, perplexity

    def inference(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, indices

    def encode(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1), flatten_idx=True)
        return zq, indices

    def decode(self, indices):
        z = self.codebook.lookup(indices)
        return z


class Sub_Quantizer(nn.Module):
    def __init__(
            self,
            code_dim: int,
            codebook_num: int,
            codebook_size: int,
            #scodebook
    ):
        super().__init__()
        self.codebook = ResidualVQ(
            dim=code_dim,
            num_quantizers=codebook_num,
            codebook_size=codebook_size
        )
        #self.register_buffer("scodebook", scodebook)

        self.register_buffer("size",torch.tensor(random.sample(range(1024),codebook_size)))
        #self.register_buffer("transform",torch.randint(low=0,high=codebook_size,size=(1,codebook_size)))
        self.register_buffer("count",torch.zeros(size=[codebook_size]))
        self.register_buffer("pin", torch.zeros(size=[codebook_size]))





    def forward(self, z, scodebook):

        self.codebook.embed=scodebook[self.size]

        zq, indices = self.codebook.forward_index(z.transpose(2, 1))




        return zq, self.size(indices)

    def inference(self, z,scodebook,lid):
        #print(scodebook.shape)#1024 192
        #print(z.shape)#1 192 113
        if (self.count<=0).all() and (self.pin==0).all():
            self.init(scodebook)

        if (((self.count<=0) & (self.pin==0)).any()):
            self.inspect(scodebook,lid)

        zq, indices = self.codebook.forward_index(z.transpose(2, 1))

        indices=indices.squeeze(0)


        index=[i.tolist() for i in indices if self.pin[i] == 0]


        if index != []:
            try:
                self.count[index]=self.count[index]-1
            except:
                print(index)
                print(self.count)
                print(index)
                exit(99)

        zq = zq.transpose(2, 1)
        return zq, self.size[indices]

    def inspect(self,scodebook,lid):
        #if (self.count<=-900).any():
        print("检查")
        self.pin[self.count<=0]=1
        max=self.count.max()
        daichuli=(self.count==max)&(self.pin==0)
        djuzhen=self.size[daichuli]
        tzh=random.sample([x for x in range(1024) if x not in self.size],len(djuzhen))
        tzh=torch.tensor(tzh,device=scodebook.device,dtype=self.size.dtype)
        self.size[daichuli]=tzh
        self.count.fill_(500)
        self.codebook.set_codebook(scodebook[self.size])

        print(lid)
        print(self.size[self.pin == 1])
        print("待处理的长度",djuzhen)
        print("已有长度",len(self.size[self.pin == 1]))

    def init(self,scodebook):

        self.codebook.set_codebook(scodebook[self.size])
        self.count.fill_(500)



    def encode(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1), flatten_idx=True)
        return zq, indices

    def decode(self, indices):
        z = self.codebook.lookup(indices)
        return z
