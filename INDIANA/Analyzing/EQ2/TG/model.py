import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import binary_search
from utils import print_hms
import time
import numpy as np
from tqdm import tqdm
import pickle
import random

random_seed = 1024
random.seed(random_seed)
torch.manual_seed(random_seed)
class GCRNN(nn.Module):
    def __init__(self, user_num, comp_num, rel_num, emb_dim, user_id_max, cuda):
        super(GCRNN, self).__init__()
        self.device0 = torch.device(cuda)
        print("Utilizing", self.device0)
        self.user_num = user_num
        self.comp_num = comp_num
        self.entity_num = user_num + comp_num + 2
        self.ent_embedding_layer = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0) # Init User/POI embeddings (u_x(t_1)^0, p_y(t_1)^0)
        self.c0_embedding_layer_u = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0) # for cell state in LSTM
        self.rel_embedding_layer = nn.Embedding(rel_num, emb_dim, sparse = False).to(self.device0) # Cw
        self.rel_num = rel_num
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0) # RNN_u
        self.POI_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0) # RNN_p
        self.user_id_max = user_id_max
        print("Xavier_Normalization")
        nn.init.xavier_normal_(self.ent_embedding_layer.weight.data)
        nn.init.xavier_normal_(self.c0_embedding_layer_u.weight.data)
        nn.init.xavier_normal_(self.rel_embedding_layer.weight.data)

    def forward(self, user_batch, comp_batch, job_batch, start_batch, g, splitted_g, history_length, remove_list):
        seed_list = []
        seed_entid = []
        train_t = []
        comp_target = []
        for comp_list in comp_batch:
            comp_target.append(comp_list)
        job_target = []
        for job_list in job_batch:
            job_target.append(job_list)
        for time_list, user in zip(start_batch, user_batch):
            for time in time_list:
                train_t.append(time)
                seed_entid.append(user)
        latest_train_time = max(train_t)
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user in zip(start_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)
        
        #Start KG-RNN
        ent_embs = self.seq_GCRNN_batch(g, splitted_g, latest_train_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), sorted = True, return_inverse = True)
        user_embs = ent_embs[index_for_ent_emb]
        u_time_embs = user_embs

        # target_c_embs, all_c_embs = POI embeddings
        target_c_embs = self.ent_embedding_layer(torch.cat(comp_target).to(self.device0)) # (N, emb_dim)
        all_c_embs = self.ent_embedding_layer(torch.tensor(list(range(self.comp_num))).to(self.device0) + self.user_id_max + 1) # (POI_num, emb_dim)

        # pos = positive
        pos_score_comp = torch.sum(u_time_embs * target_c_embs, 1).unsqueeze(1) # (batch, 1)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1,0)) # (N, POI_num)

        comp_loss_procedure = pos_score_comp - torch.logsumexp(all_score_comp, 1).unsqueeze(1)
        comp_NLL_loss = -torch.sum(comp_loss_procedure)

        return comp_NLL_loss

    def inference(self, user_batch, test_time_batch, g, splitted_g, history_length, remove_list):
        seed_list = []
        seed_entid = []
        test_t = []
        for test_time, user in zip(test_time_batch, user_batch):
            test_t.append(test_time)
            seed_entid.append(user)
        latest_test_time = max(test_t)
        for i in range(latest_test_time+1):
            seed_list.append(set())
        for test_time, user in zip(test_time_batch, user_batch):
            seed_list[test_time].add(user)

        ent_embs = self.seq_GCRNN_batch(g, splitted_g, latest_test_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_test_time + torch.tensor(test_t), sorted = True, return_inverse = True)
        u_time_embs = ent_embs[index_for_ent_emb]

        all_c_embs = self.ent_embedding_layer(torch.tensor(list(range(self.comp_num))).to(self.device0) + self.user_id_max + 1)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1,0))


        return all_score_comp


    def msg_GCN(self,edges):  # msg function for GNN
        return {'m' : edges.src['node_emb']}

    def reduce_GCN(self,nodes): # reduce function for GNN
        return {'node_emb2': nodes.mailbox['m'].mean(1)}

    def update_node(self,nodes):
        return {'node_emb': nodes.data['node_emb'] + nodes.data['node_emb2']}
        
    def seq_GCRNN_batch(self, g, splitted_g, latest_train_time, seed_list, history_length, remove_list):
        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        a2 = time.time()
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num + self.comp_num)
        for i in range(latest_train_time, -1, -1): # Preparing KG-RNN's chronological input, I-TKG
            check_lifetime[list(seed_list[i])] = history_length # we do not use this on INDIANA
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            hop1_u, hop1_v = splitted_g[i].in_edges(v = list(future_needed_nodes), form = 'uv')
            hop1_u = hop1_u.to(self.device0)
            hop1_v = hop1_v.to(self.device0)
            gcn_seed_per_time.append(list(future_needed_nodes))
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v)) # Seed's Edge
            check_lifetime[check_lifetime>0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - remove_list[i-1] - set(np.where(check_lifetime==0)[0]) # seed next
            except:
                pass
        self.rel_embedding = self.rel_embedding_layer(torch.tensor(range(self.rel_num)).to(self.device0))
        g = g.to(self.device0)
        g.ndata['node_emb'] = self.ent_embedding_layer(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        entity_embs = []
        entity_index = []
        for i in range(latest_train_time+1): # 0 -> latest, KG-RNN start from the first time-window
            inverse = latest_train_time -i
            if len(gcn_seed_per_time[inverse]) > 0:
                changed = sorted(gcn_seed_per_time[inverse])
                thresh = binary_search(changed, self.user_id_max + 1)
                user_seed_ = changed[:thresh]
                user_seed_ = changed
                user_prev_hn = g.ndata['node_emb'][user_seed_]
                user_prev_cn = g.ndata['cx'][user_seed_]
                edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse], message_func = self.msg_GCN, reduce_func = self.reduce_GCN)
                if edge_num > 0:
                    g.ndata['node_emb'] = g.ndata['node_emb2'] + g.ndata['node_emb']
                    g.ndata.pop('node_emb2')
                user_input = g.ndata['node_emb'][user_seed_]
                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))
                g.ndata['node_emb'][user_seed_] = user_hn
                g.ndata['cx'][user_seed_] = user_cn
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i
                entity_embs.append(seed_emb)
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))
        entity_embs = torch.cat(entity_embs).to(self.device0)
        entity_index = torch.cat(entity_index)
        return entity_embs[entity_index.argsort()]

