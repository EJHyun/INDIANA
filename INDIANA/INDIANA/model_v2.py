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
    def __init__(self, user_num, comp_num, rel_num, emb_dim, user_id_max, cuda,
                 neg_sampling=False, num_neg=10,
                 poi_temporal=False, separate_poi_rnn=False):
        super(GCRNN, self).__init__()
        self.device0 = torch.device(cuda)
        print("Utilizing", self.device0)
        self.user_num = user_num
        self.comp_num = comp_num
        self.entity_num = user_num + comp_num + 2
        self.emb_dim = emb_dim

        # ------------------------------------------------------------------
        # Option flags. Defaults match the implementation used to produce
        #   neg_sampling     : K negative samples per positive   vs full softmax over all POIs
        #   poi_temporal     : KG-RNN-propagated POI emb (p_tilde_y) vs static POI emb p_y
        #   separate_poi_rnn : separate LSTM for POI side (GRNN_p) vs share user LSTM 
        # ------------------------------------------------------------------
        self.neg_sampling = neg_sampling
        self.num_neg = num_neg
        self.poi_temporal = poi_temporal
        self.separate_poi_rnn = separate_poi_rnn

        self.ent_embedding_layer = nn.Embedding(self.entity_num, emb_dim, sparse=False).to(self.device0)  # Init User/POI embeddings
        self.c0_embedding_layer_u = nn.Embedding(self.entity_num, emb_dim, sparse=False).to(self.device0)  # for cell state in LSTM
        self.rel_embedding_layer = nn.Embedding(rel_num, emb_dim, sparse=False).to(self.device0)  # Cw
        self.rel_num = rel_num

        # User-side LSTM (also used for POI side when separate_poi_rnn=False).
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias=True).to(self.device0)  # RNN_u
        # POI-side LSTM (RNN_p). Instantiated only when separate_poi_rnn=True.
        if self.separate_poi_rnn:
            self.POI_RNN = nn.LSTMCell(emb_dim, emb_dim, bias=True).to(self.device0)
        else:
            self.POI_RNN = None

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
        for i in range(latest_train_time + 1):
            seed_list.append(set())
        for time_list, user in zip(start_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)

        # Start KG-RNN. This updates g.ndata['node_emb'] in place and returns time-aligned user states.
        ent_embs = self.seq_GCRNN_batch(g, splitted_g, latest_train_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), sorted=True, return_inverse=True)
        user_embs = ent_embs[index_for_ent_emb]
        u_time_embs = user_embs

        target_idx = torch.cat(comp_target).to(self.device0)

        if self.poi_temporal:
            target_c_embs = g.ndata['node_emb'][target_idx]
        else:
            target_c_embs = self.ent_embedding_layer(target_idx)

        all_poi_global_idx = torch.arange(self.comp_num, device=self.device0) + self.user_id_max + 1
        if self.poi_temporal:
            all_c_embs = g.ndata['node_emb'][all_poi_global_idx]
        else:
            all_c_embs = self.ent_embedding_layer(all_poi_global_idx)

        pos_score_comp = torch.sum(u_time_embs * target_c_embs, 1).unsqueeze(1)  # (batch, 1)

        if self.neg_sampling:
            n_pos = pos_score_comp.size(0)
            neg_local_idx = torch.randint(0, self.comp_num, (n_pos, self.num_neg), device=self.device0)  # POI local indices in [0, comp_num)
            neg_embs = all_c_embs[neg_local_idx]  # (N, K, emb_dim)
            neg_score = torch.bmm(u_time_embs.unsqueeze(1), neg_embs.transpose(1, 2)).squeeze(1)  # (N, K)
            logits = torch.cat([pos_score_comp, neg_score], dim=1)  # (N, 1+K)
            comp_loss_procedure = pos_score_comp.squeeze(1) - torch.logsumexp(logits, dim=1)
            comp_NLL_loss = -torch.sum(comp_loss_procedure)
        else:
            all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1, 0))  # (N, POI_num)
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
        for i in range(latest_test_time + 1):
            seed_list.append(set())
        for test_time, user in zip(test_time_batch, user_batch):
            seed_list[test_time].add(user)

        ent_embs = self.seq_GCRNN_batch(g, splitted_g, latest_test_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_test_time + torch.tensor(test_t), sorted=True, return_inverse=True)
        u_time_embs = ent_embs[index_for_ent_emb]

        all_poi_global_idx = torch.arange(self.comp_num, device=self.device0) + self.user_id_max + 1
        if self.poi_temporal:
            all_c_embs = g.ndata['node_emb'][all_poi_global_idx]
        else:
            all_c_embs = self.ent_embedding_layer(all_poi_global_idx)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1, 0))

        return all_score_comp

    def msg_GCN(self, edges):  # msg function for KGNN
        return {'m': edges.src['node_emb'] * self.rel_embedding[edges.data['cat_id'].type(torch.LongTensor)]}

    def reduce_GCN(self, nodes):  # reduce function for KGNN
        return {'node_emb2': nodes.mailbox['m'].mean(1)}

    def update_node(self, nodes):
        return {'node_emb': nodes.data['node_emb'] + nodes.data['node_emb2']}

    def seq_GCRNN_batch(self, g, splitted_g, latest_train_time, seed_list, history_length, remove_list):
        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        a2 = time.time()
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num + self.comp_num)
        for i in range(latest_train_time, -1, -1):  # Preparing KG-RNN's chronological input, I-TKG
            check_lifetime[list(seed_list[i])] = history_length  # we do not use this on INDIANA
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            hop1_u, hop1_v = splitted_g[i].in_edges(v=list(future_needed_nodes), form='uv')
            hop1_u = hop1_u.to(self.device0)
            hop1_v = hop1_v.to(self.device0)
            gcn_seed_per_time.append(list(future_needed_nodes))
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v))  # Seed's Edge
            check_lifetime[check_lifetime > 0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - remove_list[i - 1] - set(np.where(check_lifetime == 0)[0])  # seed next
            except:
                pass
        self.rel_embedding = self.rel_embedding_layer(torch.tensor(range(self.rel_num)).to(self.device0))
        g = g.to(self.device0)
        g.ndata['node_emb'] = self.ent_embedding_layer(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        entity_embs = []
        entity_index = []
        for i in range(latest_train_time + 1):  # 0 -> latest, KG-RNN start from the first time-window
            inverse = latest_train_time - i
            if len(gcn_seed_per_time[inverse]) > 0:
                changed = sorted(gcn_seed_per_time[inverse])
                thresh = binary_search(changed, self.user_id_max + 1)
                user_seed_ = changed[:thresh]
                user_seed_ = changed 
                user_prev_hn = g.ndata['node_emb'][user_seed_]
                user_prev_cn = g.ndata['cx'][user_seed_]
                edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                g.send_and_recv(edges=gcn_seed_1hopedge_per_time[inverse], message_func=self.msg_GCN, reduce_func=self.reduce_GCN)
                if edge_num > 0:
                    g.ndata['node_emb'] = g.ndata['node_emb2'] + g.ndata['node_emb']
                    g.ndata.pop('node_emb2')
                user_input = g.ndata['node_emb'][user_seed_]

                if self.separate_poi_rnn:
                    seed_tensor = torch.tensor(user_seed_, device=self.device0)
                    is_user = seed_tensor <= self.user_id_max
                    is_poi = ~is_user

                    user_hn = torch.empty_like(user_input)
                    user_cn = torch.empty_like(user_input)

                    if is_user.any():
                        hn_u, cn_u = self.user_RNN(
                            user_input[is_user],
                            (user_prev_hn[is_user], user_prev_cn[is_user]),
                        )
                        user_hn[is_user] = hn_u
                        user_cn[is_user] = cn_u
                    if is_poi.any():
                        hn_p, cn_p = self.POI_RNN(
                            user_input[is_poi],
                            (user_prev_hn[is_poi], user_prev_cn[is_poi]),
                        )
                        user_hn[is_poi] = hn_p
                        user_cn[is_poi] = cn_p
                else:
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
