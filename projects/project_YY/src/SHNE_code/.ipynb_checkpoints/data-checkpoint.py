# Parts of this code is borrowed from the original SHNE paper: https://github.com/chuxuzhang/WSDM2019_SHNE
import six.moves.cPickle as pickle
#import pandas as pd
import numpy as np
import string
import re
import os
import random
#from keras.preprocessing import sequence
from itertools import *

class input_data():
    def __init__(self, args):
        self.args = args

        def load_p_content(path, word_n = 3070397):
            f = open(path, 'rb')
            p_content_set = pickle.load(f)
            f.close()
            def remove_unk(x):
                return [[1 if w >= word_n else w for w in sen] for sen in x]

            p_content, p_content_id = p_content_set
            p_content = remove_unk(p_content)

            # padding with max len 
            for i in range(len(p_content)):
                if len(p_content[i]) > self.args.c_len:
                    p_content[i] = p_content[i][:self.args.c_len]
                else:
                    pad_len = self.args.c_len - len(p_content[i])
                    p_content[i] = np.lib.pad(p_content[i], (0, pad_len), 'constant', constant_values=(0,0))

            # for i in range(len(p_content)):
            # 	if len(p_content[i]) < self.args.c_len:
            # 		print i
            # 		print p_content[i]
            # 		break

            p_content_set = (p_content, p_content_id)

            return p_content_set

        def load_word_embed(path, word_n = 3070397, word_dim = 100):
            with open(path) as f:
                first_line = f.readline()
                word_n = int(first_line.split(" ")[0])
                print(word_n)
            word_embed = np.zeros((word_n + 2, word_dim))
#             print("word_embed",word_embed.shape)
            f = open(path,'r')
            for line in islice(f, 1, None):
#                 print(line)
                index = int(line.split()[0])
                embed = np.array(line.split()[1:])
                word_embed[index] = embed

            return word_embed

        content_path=os.path.join(self.args.datapath, self.args.content_filename)
        embeddings_path=os.path.join(self.args.datapath, self.args.embeddings_filename)
        self.p_content, self.p_content_id = load_p_content(path = content_path)
        self.word_embed = load_word_embed(path = embeddings_path, word_dim=self.args.embed_d)

        self.quad_sample_p = self.compute_sample_p()
        #print self.total_quad_n


    def compute_sample_p(self):
        print("computing sampling ratio for each kind of quad ...")
        window = self.args.win_s
        walk_L = self.args.walk_l
        A_n = self.args.A_n
        P_n = self.args.P_n
        V_n = self.args.V_n
        B_n = self.args.B_n

        total_quad_n = [0.0] * 16 # nine kinds of quads
        het_walk_f = open(self.args.m2v_walk, "r")
        centerNode = ''
        neighNode = ''

        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            if len(path) < window:
                continue #if paths are incomplete!
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_quad_n[0] += 1
                                elif neighNode[0] == 'p':
                                    total_quad_n[1] += 1
                                elif neighNode[0] == 'v':
                                    total_quad_n[2] += 1
                                elif neighNode[0] == 'b':
                                    total_quad_n[3] += 1
                    elif centerNode[0]=='p':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_quad_n[4] += 1
                                elif neighNode[0] == 'p':
                                    total_quad_n[5] += 1
                                elif neighNode[0] == 'v':
                                    total_quad_n[6] += 1
                                elif neighNode[0] == 'b':
                                    total_quad_n[7] += 1
                    elif centerNode[0]=='v':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_quad_n[8] += 1
                                elif neighNode[0] == 'p':
                                    total_quad_n[9] += 1
                                elif neighNode[0] == 'v':
                                    total_quad_n[10] += 1
                                elif neighNode[0] == 'b':
                                    total_quad_n[11] += 1
                    elif centerNode[0]=='b':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_quad_n[12] += 1
                                elif neighNode[0] == 'p':
                                    total_quad_n[13] += 1
                                elif neighNode[0] == 'v':
                                    total_quad_n[14] += 1
                                elif neighNode[0] == 'b':
                                    total_quad_n[15] += 1
        het_walk_f.close()

        for i in range(len(total_quad_n)):
            ######THIS SECTION COULD BE ISSUE
            try:

                total_quad_n[i] = self.args.batch_s / total_quad_n[i]
            except:
                total_quad_n[i] = 1
            #############
        print("sampling ratio computing finish.")

        return total_quad_n


    def gen_het_walk_quad_all(self):
#         print ("sampling quad relations ...")
        quad_list_all = [[] for k in range(16)] # 16 kinds of quads
        window = self.args.win_s
        walk_L = self.args.walk_l
        A_n = self.args.A_n
        P_n = self.args.P_n
        V_n = self.args.V_n
        B_n = self.args.B_n
        quad_sample_p = self.quad_sample_p # use sampling to avoid memory explosion

        het_walk_f = open(self.args.m2v_walk, "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            if len(path) < window:
                continue
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < quad_sample_p[0]:
                                    negNode = random.randint(0, A_n - 1)
                                    # random negative sampling get similar performance as noise distribution sampling
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[0].append(quad)
                                elif neighNode[0] == 'p' and random.random() < quad_sample_p[1]:
                                    negNode = random.randint(0, P_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[1].append(quad)
                                elif neighNode[0] == 'v' and random.random() < quad_sample_p[2]:
                                    negNode = random.randint(0, V_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[2].append(quad)
                                elif neighNode[0] == 'b' and random.random() < quad_sample_p[3]:
                                    negNode = random.randint(0, B_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[3].append(quad)
                                    
                    elif centerNode[0]=='p':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < quad_sample_p[4]:
                                    negNode = random.randint(0, A_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[4].append(quad)
                                elif neighNode[0] == 'p' and random.random() < quad_sample_p[5]:
                                    negNode = random.randint(0, P_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[5].append(quad)
                                elif neighNode[0] == 'v' and random.random() < quad_sample_p[6]:
                                    negNode = random.randint(0, V_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[6].append(quad)
                                elif neighNode[0] == 'b' and random.random() < quad_sample_p[7]:
                                    negNode = random.randint(0, B_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[7].append(quad)
                                    
                    elif centerNode[0]=='v':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < quad_sample_p[8]:
                                    negNode = random.randint(0, A_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[8].append(quad)
                                elif neighNode[0] == 'p' and random.random() < quad_sample_p[9]:
                                    negNode = random.randint(0, P_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[9].append(quad)
                                elif neighNode[0] == 'v' and random.random() < quad_sample_p[10]:
                                    negNode = random.randint(0, V_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[10].append(quad)
                                elif neighNode[0] == 'b' and random.random() < quad_sample_p[11]:
                                    negNode = random.randint(0, B_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[11].append(quad)
                                    
                    elif centerNode[0]=='b':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a' and random.random() < quad_sample_p[12]:
                                    negNode = random.randint(0, A_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[12].append(quad)
                                elif neighNode[0] == 'p' and random.random() < quad_sample_p[13]:
                                    negNode = random.randint(0, P_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[13].append(quad)
                                elif neighNode[0] == 'v' and random.random() < quad_sample_p[14]:
                                    negNode = random.randint(0, V_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[14].append(quad)
                                elif neighNode[0] == 'b' and random.random() < quad_sample_p[15]:
                                    negNode = random.randint(0, B_n - 1)
                                    quad = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    quad_list_all[15].append(quad)
        het_walk_f.close()

        return quad_list_all



