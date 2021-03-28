# This code is borrowed from the original SHNE paper: https://github.com/chuxuzhang/WSDM2019_SHNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import re

class SHNE_Encoder(nn.Module):
    def __init__(self, args, p_content, word_embed):
        super(SHNE_Encoder, self).__init__()
        self.args = args
        word_embed = torch.FloatTensor(word_embed)
        #load pre-trained word embeddings
        self.word_embedding = nn.Embedding.from_pretrained(word_embed)
        self.p_content = torch.LongTensor(p_content)

        if args.cuda:
            self.word_embedding = self.word_embedding.cuda()
            self.p_content = self.p_content.cuda()

        self.a_latent = torch.nn.Parameter(torch.ones(args.A_n, args.embed_d), requires_grad=True)
        self.p_latent = torch.nn.Parameter(torch.ones(args.P_n, args.embed_d), requires_grad=True)
        self.v_latent = torch.nn.Parameter(torch.ones(args.V_n, args.embed_d), requires_grad=True)
        self.b_latent = torch.nn.Parameter(torch.ones(args.B_n, args.embed_d), requires_grad=True) #blocks
        self.rnn_p = nn.LSTM(args.embed_d, args.embed_d, 1, bidirectional = False)

        init.xavier_normal_(self.a_latent)
        init.xavier_normal_(self.p_latent)
        init.xavier_normal_(self.v_latent)
        init.xavier_normal_(self.b_latent)


    def a_encode(self, id_batch):
        return self.a_latent[id_batch]


    def v_encode(self, id_batch):
        return self.v_latent[id_batch]


    def b_encode(self, id_batch):
        return self.b_latent[id_batch]


    def p_encode(self, id_batch):
        p_content_batch = torch.LongTensor(len(id_batch), self.args.c_len)
        for i in range(len(id_batch)):
            if len(self.p_content[id_batch[i]]) < len(id_batch): continue

            p_content_batch[i] = self.p_content[id_batch[i]]
        p_content_batch = p_content_batch.view(1, -1)

        if self.args.cuda:
            p_content_batch = p_content_batch.cuda()

        # load pretrain word embeddings
        p_e_pretrain = self.word_embedding(p_content_batch)
        p_e_pretrain = p_e_pretrain.view(len(id_batch), -1, self.args.embed_d)

        # LSTM encoder 
        p_e_pretrain = torch.transpose(p_e_pretrain, 0, 1)
        p_deep_latent, last_state = self.rnn_p(p_e_pretrain)
        p_deep_latent = torch.mean(p_deep_latent, 0)
        return p_deep_latent

#Braden: This section is interesting plus the saved node embeddings
    def encode_all(self, quad_batch, quad_index):
        c_id_batch = [x[0] for x in quad_batch]
        pos_id_batch = [x[1] for x in quad_batch]
        neg_id_batch = [x[2] for x in quad_batch]
        b_id_batch = [x[3] for x in quad_batch]
        c_embed = []
        pos_embed = []
        neg_embed = []
        b_embed = []

        if quad_index == 0: # a-a-a-a 
            c_embed = self.a_encode(c_id_batch)
            pos_embed = self.a_encode(pos_id_batch)
            neg_embed = self.a_encode(neg_id_batch)
            b_embed = self.a_encode(b_id_batch)
        elif quad_index == 1: # a-p-p-p
            c_embed = self.a_encode(c_id_batch)
            pos_embed = self.p_encode(pos_id_batch)
            neg_embed = self.p_encode(neg_id_batch)
            b_embed = self.p_encode(b_id_batch)
        elif quad_index == 2: # a-v-v-v
            c_embed = self.a_encode(c_id_batch)
            pos_embed = self.v_encode(pos_id_batch)
            neg_embed = self.v_encode(neg_id_batch)
            b_embed = self.v_encode(b_id_batch)
        elif quad_index == 3: # p-a-a-a
            c_embed = self.p_encode(c_id_batch)
            pos_embed = self.a_encode(pos_id_batch)
            neg_embed = self.a_encode(neg_id_batch)
            b_embed = self.a_encode(b_id_batch)
        elif quad_index == 4: # p-p-p-p
            c_embed = self.p_encode(c_id_batch)
            pos_embed = self.p_encode(pos_id_batch)
            neg_embed = self.p_encode(neg_id_batch)
            b_embed = self.p_encode(b_id_batch)
        elif quad_index == 5: # p-v-v-v
            c_embed = self.p_encode(c_id_batch)
            pos_embed = self.v_encode(pos_id_batch)
            neg_embed = self.v_encode(neg_id_batch)
            b_embed = self.v_encode(b_id_batch)
        elif quad_index == 6: # v-a-a-a
            c_embed = self.v_encode(c_id_batch)
            pos_embed = self.a_encode(pos_id_batch)
            neg_embed = self.a_encode(neg_id_batch)
            b_embed = self.a_encode(b_id_batch)
        elif quad_index == 7: # v-p-p-p
            c_embed = self.v_encode(c_id_batch)
            pos_embed = self.p_encode(pos_id_batch)
            neg_embed = self.p_encode(neg_id_batch)
            b_embed = self.p_encode(b_id_batch)
        elif quad_index == 8: # v-v-v-v
            c_embed = self.v_encode(c_id_batch)
            pos_embed = self.v_encode(pos_id_batch)
            neg_embed = self.v_encode(neg_id_batch)
            b_embed = self.v_encode(b_id_batch)
        elif quad_index == 9: # a-b-b-b
            c_embed = self.a_encode(c_id_batch)
            pos_embed = self.b_encode(pos_id_batch)
            neg_embed = self.b_encode(neg_id_batch)
            b_embed = self.b_encode(b_id_batch)
        elif quad_index == 10: # p-b-b-b
            c_embed = self.p_encode(c_id_batch)
            pos_embed = self.b_encode(pos_id_batch)
            neg_embed = self.b_encode(neg_id_batch)
            b_embed = self.b_encode(b_id_batch)
        elif quad_index == 11: # v-b-b-b
            c_embed = self.v_encode(c_id_batch)
            pos_embed = self.b_encode(pos_id_batch)
            neg_embed = self.b_encode(neg_id_batch)
            b_embed = self.b_encode(b_id_batch)
        elif quad_index == 12: # b-b-b-b
            c_embed = self.b_encode(c_id_batch)
            pos_embed = self.b_encode(pos_id_batch)
            neg_embed = self.b_encode(neg_id_batch)
            b_embed = self.b_encode(b_id_batch)
        elif quad_index == 13: # b-a-a-a
            c_embed = self.b_encode(c_id_batch)
            pos_embed = self.a_encode(pos_id_batch)
            neg_embed = self.a_encode(neg_id_batch)
            b_embed = self.a_encode(b_id_batch)
        elif quad_index == 14: # b-p-p-p
            c_embed = self.b_encode(c_id_batch)
            pos_embed = self.p_encode(pos_id_batch)
            neg_embed = self.p_encode(neg_id_batch)
            b_embed = self.p_encode(b_id_batch)
        elif quad_index == 15: # b-v-v-v
            c_embed = self.b_encode(c_id_batch)
            pos_embed = self.v_encode(pos_id_batch)
            neg_embed = self.v_encode(neg_id_batch)
            b_embed = self.v_encode(b_id_batch)
        elif quad_index == 16: # save embeddings for evaluation
            embed_file = open(self.args.datapath + "node_embedding.txt", "w")
            batch_s = 1000
            embed_d = self.args.embed_d
            for i in range(3):
                if i == 0:
                    batch_number = int(self.args.A_n / batch_s)
                    for kk in range(batch_number): 
                        id_batch = np.arange(kk * batch_s, (kk + 1) * batch_s)
                        a_out_temp = self.a_encode(id_batch)
                        a_out_temp = a_out_temp.data.cpu().numpy()
                        for ll in range(len(id_batch)):
                            index = id_batch[ll]
                            embed_file.write('a' + str(index) + " ")
                            for j in range(embed_d - 1):
                                embed_file.write(str(a_out_temp[ll][j]) + " ")
                            embed_file.write(str(a_out_temp[ll][-1]) + "\n")

                    id_batch = np.arange(batch_number * batch_s, self.args.A_n)
                    a_out_temp = self.a_encode(id_batch)
                    a_out_temp = a_out_temp.data.cpu().numpy()
                    for ll in range(len(id_batch)):
                        index = id_batch[ll]
                        embed_file.write('a' + str(index) + " ")
                        for j in range(embed_d - 1):
                            embed_file.write(str(a_out_temp[ll][j]) + " ")
                        embed_file.write(str(a_out_temp[ll][-1]) + "\n")
                elif i == 1:
                    batch_number = int(self.args.P_n / batch_s)
                    for kk in range(batch_number): 
                        id_batch = np.arange(kk * batch_s, (kk + 1) * batch_s)
                        p_out_temp = self.p_encode(id_batch)
                        p_out_temp = p_out_temp.data.cpu().numpy()
                        for ll in range(len(id_batch)):
                            index = id_batch[ll]
                            embed_file.write('p' + str(index) + " ")
                            for j in range(embed_d - 1):
                                embed_file.write(str(p_out_temp[ll][j]) + " ")
                            embed_file.write(str(p_out_temp[ll][-1]) + "\n")

                    id_batch = np.arange(batch_number * batch_s, self.args.P_n)
                    p_out_temp = self.p_encode(id_batch)
                    p_out_temp = p_out_temp.data.cpu().numpy()
                    for ll in range(len(id_batch)):
                            index = id_batch[ll]
                            embed_file.write('p' + str(index) + " ")
                            for j in range(embed_d - 1):
                                embed_file.write(str(p_out_temp[ll][j]) + " ")
                            embed_file.write(str(p_out_temp[ll][-1]) + "\n")
                elif i == 2:
                    id_batch = np.arange(self.args.V_n)
                    v_out = self.v_encode(id_batch)
                    v_out = v_out.data.cpu().numpy()
                    for ll in range(len(v_out)):
                        index = id_batch[ll]
                        embed_file.write('v' + str(index) + " ")
                        for j in range(embed_d - 1):
                            embed_file.write(str(v_out[ll][j]) + " ")
                        embed_file.write(str(v_out[ll][-1]) + "\n")
                else:
                    id_batch = np.arange(self.args.B_n)
                    b_out = self.b_encode(id_batch)
                    b_out = b_out.data.cpu().numpy()
                    for ll in range(len(b_out)):
                        index = id_batch[ll]
                        embed_file.write('b' + str(index) + " ")
                        for j in range(embed_d - 1):
                            embed_file.write(str(b_out[ll][j]) + " ")
                        embed_file.write(str(b_out[ll][-1]) + "\n")
            embed_file.close()

            return [], [], [], []

        return c_embed, pos_embed, neg_embed, b_embed


    def forward(self, quad_batch, quad_index):
        if quad_index == 9:
            c_out, p_out, n_out, b_out = self.encode_all(quad_batch, quad_index)
            return c_out, p_out, n_out, b_out
        else:
            c_out, p_out, n_out, b_out  = self.encode_all(quad_batch, quad_index)
            return c_out, p_out, n_out, b_out


def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, b_embed_batch, embed_d):
    # cross entropy loss/ skip-gram with negative sampling (negative size = 1)
    batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]

    c_embed = c_embed_batch.view(batch_size, 1, embed_d)
    pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
    neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)
    b_embed = b_embed_batch.view(batch_size, embed_d, 1)

    out_p = torch.bmm(c_embed, pos_embed)
    out_n = - torch.bmm(c_embed, neg_embed)
    out_b = - torch.bmm(c_embed, b_embed)

    sum_p = F.logsigmoid(out_p)
    sum_n = F.logsigmoid(out_n)
    sum_b = F.logsigmoid(out_b)
    loss_sum = - (sum_p + sum_n + sum_b)

    loss_sum = loss_sum.sum() / batch_size

    return loss_sum


def evaluation_test(datapath, node_n, embed_d):
    node_embed = np.around(np.random.normal(0, 0.01, [node_n, embed_d]), 4)
    embed_file = open(datapath + "node_embedding.txt", "r")
    for line in embed_file:
        line = line.strip()
        node_index = re.split(' ',line)[0]
        if len(node_index) and (node_index[0] == 'a' or node_index[0] == 'p' or node_index[0] == 'v'):
            index_label = node_index[0]
            index_id = int(node_index[1:])
            embed_array = re.split(' ',line)[1:]
            coefs = np.asarray(embed_array, dtype='float32')
            if index_label == 'v':
                node_embed[index_id] = coefs
    embed_file.close()		

    score_1 = np.dot(node_embed[9], node_embed[13]) # KDD vs WSDM
    score_2 = np.dot(node_embed[9], node_embed[17])	# KDD vs CVPR

    print ("similarity score between KDD and WSDM: " + str(score_1))
    print ("similarity score between KDD and CVPR: " + str(score_2))


#evaluation_test("../data/", 18, 128)


