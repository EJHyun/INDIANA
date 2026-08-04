import torch
import pickle
import time
import random
import argparse
from utils import print_metrics, METRIC_KEYS
from dgl.data.utils import load_graphs
import model as model
import numpy as np
import dgl

parser = argparse.ArgumentParser(description="Intention-Aware Next-POI Recommendation (INDIANA)")
parser.add_argument("--dataset", default="NYC", type=str, help="NYC, TKY or CA")
parser.add_argument("--data_dir", default=".", type=str, help="directory holding the preprocessed pickles and .TKG")
parser.add_argument("--device", default="cuda:0", type=str, help="Which device do you wanna use")
parser.add_argument("--lr", default=0.001, type=float, help="")
parser.add_argument("--emb_dim", default=0, type=int, help="Embedding dimension dim, 0 selects the value reported in the paper: 100 for NYC, 150 for TKY and CA")
parser.add_argument("--num_layers", default=2, type=int, help="J, the number of KGNN layers")
parser.add_argument("--bptt", default=200, type=int, help="snapshots between optimizer steps")
parser.add_argument("--epochs", default=300, type=int, help="")
parser.add_argument("--seed", default=1024, type=int, help="")
parser.add_argument("--rnn_steps", default="active", type=str, choices=["active", "all", "poi_all"],
                    help="active advances GRNN only at snapshots where the entity is checked in; all advances every entity at every snapshot")
parser.add_argument("--poi_state", default="temporal", type=str, choices=["temporal", "static"],
                    help="temporal scores with the KG-RNN state p_tilde_y(t_z) as in Eq.10; static scores with the free POI embedding")
parser.add_argument("--rel_init", default="ones", type=str, choices=["xavier", "ones"],
                    help="ones starts the elementwise relation product at identity")
parser.add_argument("--rnn_init", default="identity", type=str, choices=["default", "passthrough", "identity"],
                    help="passthrough biases input/forget gates open; identity additionally passes the input through the g gate")
parser.add_argument("--emb_std", default=0.5, type=float,
                    help="entity embedding init std, 0 keeps xavier")
parser.add_argument("--gate_bias", default=1.2, type=float,
                    help="input/forget gate bias for passthrough and identity rnn_init")
parser.add_argument("--train_scope", default="all", type=str, choices=["all", "rel", "emb", "rnn"],
                    help="which parameter groups the optimizer updates")
parser.add_argument("--fg_bias", default=None, type=float, help="forget gate bias, overrides gate_bias")
parser.add_argument("--in_bias", default=None, type=float, help="input gate bias, overrides gate_bias")
parser.add_argument("--norm", default="mean", type=str, choices=["mean", "sqrt"],
                    help="degree normalization of Eq.7 aggregation")
parser.add_argument("--g_bias", default=0.0, type=float, help="cell candidate gate bias")
parser.add_argument("--poi_gate_bias", default=None, type=float,
                    help="separate input/forget gate bias for the POI-side GRNN")
parser.add_argument("--variant", default="indiana", type=str,
                    choices=["indiana", "baseline", "kg", "tg", "gat", "att", "gru"],
                    help="indiana is the full model; baseline/kg/tg are the Table 4 ablations; gat/att/gru are the Table 5 alternatives")
args = parser.parse_args()

dataset = args.dataset
emb_dim = args.emb_dim if args.emb_dim > 0 else (100 if dataset == "NYC" else 150)
best = {k: 0 for k in METRIC_KEYS}
best_epoch = 0
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print("dataset:", dataset, "emb_dim:", emb_dim, "J:", args.num_layers, "seed:", args.seed,
      "variant:", args.variant, "rnn_steps:", args.rnn_steps, "poi_state:", args.poi_state,
      "bptt:", args.bptt, "lr:", args.lr,
      "rel_init:", args.rel_init, "rnn_init:", args.rnn_init, "emb_std:", args.emb_std,
      "gate_bias:", args.gate_bias, "train_scope:", args.train_scope,
      "fg_bias:", args.fg_bias, "in_bias:", args.in_bias, "norm:", args.norm, "g_bias:", args.g_bias,
      "poi_gate_bias:", args.poi_gate_bias)

with open('{}/{}_train.pickle'.format(args.data_dir, dataset), 'rb') as f:
    User_cnt, POI_cnt, Cat_cnt, Time_cnt, POI_dict, cat_dict, time_dict, _, _, _ = pickle.load(f)
with open('{}/{}_test.pickle'.format(args.data_dir, dataset), 'rb') as f:
    _, _, _, POI_dict_test, _, time_dict_test, _, _, _ = pickle.load(f)
print('Loading graph...')
glist, _ = load_graphs('{}/{}_train.TKG'.format(args.data_dir, dataset), [0])
Train_Graph = glist[0]
time_id = Train_Graph.edata['time_id'].numpy()
splitted_Train_Graph = [dgl.graph(([], []), num_nodes = User_cnt + POI_cnt)]
for i in range(Time_cnt):
    splitted_Train_Graph.append(Train_Graph.edge_subgraph(np.where(time_id == i)[0], relabel_nodes = False, store_ids = True))
glist, _ = load_graphs('{}/{}_all.TKG'.format(args.data_dir, dataset), [0])
Eval_Graph = glist[0]
eval_time_id = Eval_Graph.edata['time_id'].numpy()
splitted_Eval_Graph = [dgl.graph(([], []), num_nodes = User_cnt + POI_cnt)]
for i in range(Time_cnt):
    splitted_Eval_Graph.append(Eval_Graph.edge_subgraph(np.where(eval_time_id == i)[0], relabel_nodes = False, store_ids = True))

device = torch.device(args.device)
targets_by_time = [None] * Time_cnt
for user in POI_dict:
    for p, t in zip(POI_dict[user], time_dict[user]):
        if targets_by_time[t] is None:
            targets_by_time[t] = ([], [])
        targets_by_time[t][0].append(user)
        targets_by_time[t][1].append(p - User_cnt)
targets_by_time = [None if v is None else (v[0], torch.tensor(v[1], dtype=torch.long, device=device))
                   for v in targets_by_time]

probes_by_time = [None] * Time_cnt
n_test_users = 0
for user in POI_dict:
    if len(time_dict_test[user]) == 0:
        continue
    n_test_users += 1
    for p, t in zip(POI_dict_test[user], time_dict_test[user]):
        if probes_by_time[t] is None:
            probes_by_time[t] = ([], [])
        probes_by_time[t][0].append(user)
        probes_by_time[t][1].append(p)
n_test_checkins = sum(len(v[1]) for v in probes_by_time if v is not None)
probes_by_time = [None if v is None else (v[0], torch.tensor(v[1], dtype=torch.long, device=device))
                  for v in probes_by_time]

model = model.GCRNN(User_cnt, POI_cnt, Cat_cnt*2, emb_dim, User_cnt-1, args.device,
                    args.num_layers, args.variant, 5, args.poi_state, args.rnn_steps,
                    args.rel_init, args.rnn_init, args.emb_std, args.gate_bias,
                    args.fg_bias, args.in_bias, args.norm, args.g_bias, args.poi_gate_bias)
if args.train_scope == 'rel':
    train_params = list(model.rel_embedding_layer.parameters())
elif args.train_scope == 'emb':
    train_params = list(model.ent_embedding_layer.parameters()) + list(model.c0_embedding_layer_u.parameters())
elif args.train_scope == 'rnn':
    train_params = list(model.user_RNN.parameters()) + list(model.POI_RNN.parameters())
else:
    train_params = list(model.parameters())
optimizer = torch.optim.Adam(train_params, lr = args.lr)
n_checkins = sum(len(v[1]) for v in targets_by_time if v is not None)
print("Train start:", User_cnt, "users,", n_checkins, "train check-ins,",
      n_test_checkins, "test check-ins from", n_test_users, "users,", Time_cnt, "snapshots")

model.eval()
with torch.no_grad():
    print_metrics(model.evaluate(Eval_Graph, splitted_Eval_Graph, probes_by_time, Time_cnt), None)
print("epoch -1 (untrained forward pass)")

for epoch in range(args.epochs):
    epoch_start = time.time()
    model.train()
    loss_per_checkin, steps = model.train_epoch(Train_Graph, splitted_Train_Graph, targets_by_time,
                                                optimizer, args.bptt)
    train_sec = time.time() - epoch_start
    model.eval()
    with torch.no_grad():
        metrics = print_metrics(model.evaluate(Eval_Graph, splitted_Eval_Graph, probes_by_time, Time_cnt), None)
    print("epoch {}  train_loss_per_checkin {:.5f}  steps {}  train {:.1f}s  eval {:.1f}s".format(
        epoch, loss_per_checkin, steps, train_sec, time.time() - epoch_start - train_sec))
    if metrics['MRR'] > best['MRR']:
        best_epoch = epoch
        best = metrics
    print("Best " + '  '.join('{} {:.4f}'.format(k, best[k]) for k in METRIC_KEYS) + " at epoch " + str(best_epoch))
