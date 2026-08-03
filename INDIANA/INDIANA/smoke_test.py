import argparse
import pickle

import dgl
import numpy as np
import torch

import model as model_module

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="NYC")
parser.add_argument("--data_dir", default=".")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--horizon", default=40, type=int)
args = parser.parse_args()

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
    if len(users) == 8:
        break
assert users, "no users within the smoke horizon"
n_pairs = sum(len(set(t)) for t in times)
print('smoke batch: {} users, {} (user,time) pairs, horizon {}'.format(len(users), n_pairs, args.horizon))

failed = []
for variant in ['indiana', 'baseline', 'kg', 'tg', 'gat', 'att', 'gru']:
    try:
        torch.manual_seed(0)
        net = model_module.GCRNN(User_cnt, POI_cnt, Cat_cnt * 2, 32, User_cnt - 1,
                                 args.device, 2, variant)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss = net(users, pois, times, Train_Graph, splitted)
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        opt.step()
        probes = [None] * (args.horizon + 1)
        for usr, tl in zip(users, times):
            t = tl[0]
            if probes[t] is None:
                probes[t] = ([], [])
            probes[t][0].append(usr)
            probes[t][1].append(0)
        probes = [None if v is None else (v[0], torch.tensor(v[1], dtype=torch.long, device=args.device)) for v in probes]
        with torch.no_grad():
            ranks = net.evaluate(Train_Graph, splitted, probes, args.horizon + 1)
        ok = (torch.isfinite(loss).all() and len(ranks) == len(users)
              and min(ranks) >= 1 and max(ranks) <= POI_cnt and len(grads) > 0)
        print('{:9s} loss {:12.3f}  ranks {:3d} in [{},{}]  tensors_with_grad {:2d}  {}'.format(
            variant, float(loss), len(ranks), min(ranks), max(ranks), len(grads), 'OK' if ok else 'BAD'))
        if not ok:
            failed.append(variant)
    except Exception as e:
        print('{:9s} FAILED: {}: {}'.format(variant, type(e).__name__, e))
        failed.append(variant)

print('\nRESULT:', 'all variants OK' if not failed else 'FAILED -> ' + ', '.join(failed))
