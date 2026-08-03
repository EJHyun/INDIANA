import argparse
import pickle

import dgl
import numpy as np
import torch

import model as model_module

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="NYC")
parser.add_argument("--data_dir", default=".")
parser.add_argument("--device", default="cpu")
parser.add_argument("--horizon", default=60, type=int)
parser.add_argument("--probes", default=12, type=int)
parser.add_argument("--eps", default=1e-4, type=float)
args = parser.parse_args()

torch.set_default_dtype(torch.float64)

with open('{}/{}_train.pickle'.format(args.data_dir, args.dataset), 'rb') as f:
    User_cnt, POI_cnt, Cat_cnt, Time_cnt, POI_dict, cat_dict, time_dict, _, _, _ = pickle.load(f)
glist, _ = dgl.load_graphs('{}/{}_train.TKG'.format(args.data_dir, args.dataset), [0])
Train_Graph = glist[0]
splitted = [dgl.graph(([], []), num_nodes=User_cnt + POI_cnt)]
for i in range(args.horizon + 1):
    splitted.append(Train_Graph.edge_subgraph(
        np.where(Train_Graph.edata['time_id'] == i)[0], relabel_nodes=False, store_ids=True))

users, pois, cats, times = [], [], [], []
for u in list(POI_dict.keys()):
    keep = [k for k, t in enumerate(time_dict[u]) if t <= args.horizon]
    if not keep:
        continue
    users.append(u)
    pois.append(torch.tensor([POI_dict[u][k] for k in keep]).type(torch.LongTensor))
    cats.append(torch.tensor([cat_dict[u][k] for k in keep]).type(torch.LongTensor))
    times.append([time_dict[u][k] for k in keep])
    if len(users) == 6:
        break

torch.manual_seed(0)
net = model_module.GCRNN(User_cnt, POI_cnt, Cat_cnt * 2, 8, User_cnt - 1, args.device, 1)
net = net.double()


def loss_fn():
    return net(users, pois, times, Train_Graph, splitted)


net.zero_grad()
loss_fn().backward()

targets = [('ent_embedding_layer.weight', net.ent_embedding_layer.weight),
           ('rel_embedding_layer.weight', net.rel_embedding_layer.weight),
           ('user_RNN.weight_ih', net.user_RNN.weight_ih),
           ('POI_RNN.weight_ih', net.POI_RNN.weight_ih)]

rng = np.random.RandomState(0)
worst = 0.0
print('{:28s} {:>14s} {:>14s} {:>10s}'.format('parameter', 'autograd', 'numeric', 'rel.err'))
for name, p in targets:
    flat = p.data.view(-1)
    g = p.grad.view(-1)
    idxs = rng.choice(flat.numel(), size=min(args.probes, flat.numel()), replace=False)
    for k in idxs[:3]:
        k = int(k)
        orig = flat[k].item()
        flat[k] = orig + args.eps
        with torch.no_grad():
            hi = loss_fn().item()
        flat[k] = orig - args.eps
        with torch.no_grad():
            lo = loss_fn().item()
        flat[k] = orig
        num = (hi - lo) / (2 * args.eps)
        auto = g[k].item()
        denom = max(1.0, abs(num), abs(auto))
        rel = abs(num - auto) / denom
        worst = max(worst, rel)
        print('{:28s} {:14.6f} {:14.6f} {:10.2e}'.format(name + '[' + str(k) + ']', auto, num, rel))

print('\nworst relative error: {:.3e}'.format(worst))
print('VERDICT:', 'gradients CORRECT' if worst < 1e-5 else 'gradients WRONG')
