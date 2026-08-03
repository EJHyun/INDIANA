import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCRNN(nn.Module):
    def __init__(self, user_num, comp_num, rel_num, emb_dim, user_id_max, cuda, num_layers=1,
                 variant='indiana', att_window=5, poi_state='temporal', rnn_steps='active'):
        super(GCRNN, self).__init__()
        self.device0 = torch.device(cuda)
        print("Utilizing", self.device0)
        self.user_num = user_num
        self.comp_num = comp_num
        self.entity_num = user_num + comp_num + 2
        self.num_layers = num_layers
        self.variant = variant
        self.att_window = att_window
        self.poi_state = poi_state
        self.rnn_steps = rnn_steps
        self.use_relation = variant not in ('tg', 'baseline')
        self.use_rnn = variant not in ('kg', 'baseline', 'att')
        self.ent_embedding_layer = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0)
        self.c0_embedding_layer_u = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0)
        self.rel_embedding_layer = nn.Embedding(rel_num, emb_dim, sparse = False).to(self.device0)
        self.rel_num = rel_num
        cell = nn.GRUCell if variant == 'gru' else nn.LSTMCell
        self.user_RNN = cell(emb_dim, emb_dim, bias = True).to(self.device0)
        self.POI_RNN = cell(emb_dim, emb_dim, bias = True).to(self.device0)
        if variant == 'gat':
            self.att_src = nn.Linear(emb_dim, 1, bias = False).to(self.device0)
            self.att_dst = nn.Linear(emb_dim, 1, bias = False).to(self.device0)
        if variant == 'att':
            self.learnable_query_layer = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0)
        self.user_id_max = user_id_max
        print("Xavier_Normalization")
        nn.init.xavier_normal_(self.ent_embedding_layer.weight.data)
        nn.init.xavier_normal_(self.c0_embedding_layer_u.weight.data)
        nn.init.xavier_normal_(self.rel_embedding_layer.weight.data)
        if variant == 'att':
            nn.init.xavier_normal_(self.learnable_query_layer.weight.data)

    def msg_GCN(self, edges):
        m = edges.src['node_emb']
        if self.use_relation:
            m = m * self.rel_embedding[edges.data['cat_id'].long()]
        if self.variant == 'gat':
            return {'m': m, 'e': self.att_src(m) + self.att_dst(edges.dst['node_emb'])}
        return {'m': m}

    def reduce_GCN(self, nodes):
        if self.variant == 'gat':
            alpha = F.softmax(F.leaky_relu(nodes.mailbox['e'], 0.2), dim = 1)
            return {'node_emb2': (nodes.mailbox['m'] * alpha).sum(1)}
        return {'node_emb2': nodes.mailbox['m'].mean(1)}

    def _init_state(self, g):
        self.rel_ids = torch.arange(self.rel_num, device=self.device0)
        node_ids = torch.arange(g.number_of_nodes(), device=self.device0)
        return node_ids, self.ent_embedding_layer(node_ids), self.c0_embedding_layer_u(node_ids)

    def _advance(self, g, snap, node_ids, node_emb, cx):
        su, sv = snap.edges()
        if su.numel() == 0:
            return node_emb, cx
        self.rel_embedding = self.rel_embedding_layer(self.rel_ids)
        su = su.to(self.device0)
        sv = sv.to(self.device0)
        changed = node_ids if self.rnn_steps == 'all' else torch.unique(sv)
        thresh = int((changed <= self.user_id_max).sum())
        prev_hn = node_emb[changed]
        prev_cn = cx[changed]
        for _ in range(self.num_layers):
            g.ndata['node_emb'] = node_emb
            g.send_and_recv(edges=(su, sv), message_func=self.msg_GCN, reduce_func=self.reduce_GCN)
            node_emb = node_emb.clone()
            node_emb[changed] = node_emb[changed] + g.ndata['node_emb2'][changed]
            g.ndata.pop('node_emb2')
        if self.use_rnn:
            node_input = node_emb[changed]
            node_emb = node_emb.clone()
            if self.variant == 'gru':
                user_hn = self.user_RNN(node_input[:thresh], prev_hn[:thresh])
                poi_hn = self.POI_RNN(node_input[thresh:], prev_hn[thresh:])
                node_emb[changed] = torch.cat([user_hn, poi_hn], 0)
            else:
                user_hn, user_cn = self.user_RNN(node_input[:thresh], (prev_hn[:thresh], prev_cn[:thresh]))
                poi_hn, poi_cn = self.POI_RNN(node_input[thresh:], (prev_hn[thresh:], prev_cn[thresh:]))
                cx = cx.clone()
                node_emb[changed] = torch.cat([user_hn, poi_hn], 0)
                cx[changed] = torch.cat([user_cn, poi_cn], 0)
        return node_emb, cx

    def _score(self, node_ids, node_emb, seeds):
        if self.poi_state == 'static':
            poi_emb = self.ent_embedding_layer(node_ids[self.user_id_max + 1:])
        else:
            poi_emb = node_emb[self.user_id_max + 1:]
        return torch.matmul(node_emb[seeds], poi_emb.transpose(1, 0))

    def _nll(self, score_chunks, target_chunks):
        scores = torch.cat(score_chunks)
        targets = torch.cat(target_chunks)
        pos = scores.gather(1, targets.unsqueeze(1))
        return -torch.sum(pos - torch.logsumexp(scores, 1).unsqueeze(1))

    def train_epoch(self, g, splitted_g, targets_by_time, optimizer, bptt):
        g = g.to(self.device0)
        node_ids, node_emb, cx = self._init_state(g)
        score_chunks, target_chunks = [], []
        total_loss, total_pairs, steps = 0.0, 0, 0
        for i in range(len(targets_by_time)):
            node_emb, cx = self._advance(g, splitted_g[i], node_ids, node_emb, cx)
            hit = targets_by_time[i]
            if hit is not None:
                seeds, targets = hit
                score_chunks.append(self._score(node_ids, node_emb, seeds))
                target_chunks.append(targets)
            if (i + 1) % bptt == 0 and score_chunks:
                loss = self._nll(score_chunks, target_chunks)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += float(loss)
                total_pairs += sum(t.numel() for t in target_chunks)
                steps += 1
                score_chunks, target_chunks = [], []
                node_emb = node_emb.detach()
                cx = cx.detach()
        if score_chunks:
            loss = self._nll(score_chunks, target_chunks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss)
            total_pairs += sum(t.numel() for t in target_chunks)
            steps += 1
        return total_loss / max(1, total_pairs), steps

    def evaluate(self, g, splitted_g, probes_by_time, n_snapshots):
        g = g.to(self.device0)
        node_ids, node_emb, cx = self._init_state(g)
        ranks = []
        for i in range(n_snapshots):
            node_emb, cx = self._advance(g, splitted_g[i], node_ids, node_emb, cx)
            if probes_by_time[i] is not None:
                users, targets = probes_by_time[i]
                scores = self._score(node_ids, node_emb, users)
                gathered = scores.gather(1, targets.unsqueeze(1))
                ranks.extend(((scores > gathered).sum(1) + 1).tolist())
        return ranks

    def forward(self, user_batch, comp_batch, start_batch, g, splitted_g):
        g = g.to(self.device0)
        node_ids, node_emb, cx = self._init_state(g)
        latest = max(max(t) for t in start_batch)
        by_time = [None] * (latest + 1)
        for user, pois, times in zip(user_batch, comp_batch, start_batch):
            for p, t in zip(pois.tolist(), times):
                if by_time[t] is None:
                    by_time[t] = ([], [])
                by_time[t][0].append(user)
                by_time[t][1].append(p - (self.user_id_max + 1))
        score_chunks, target_chunks = [], []
        for i in range(latest + 1):
            node_emb, cx = self._advance(g, splitted_g[i], node_ids, node_emb, cx)
            if by_time[i] is not None:
                score_chunks.append(self._score(node_ids, node_emb, by_time[i][0]))
                target_chunks.append(torch.tensor(by_time[i][1], dtype=torch.long, device=self.device0))
        return self._nll(score_chunks, target_chunks)
