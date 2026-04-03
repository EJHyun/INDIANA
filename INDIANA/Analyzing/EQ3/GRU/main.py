import torch
import pickle
import time
import random
from tqdm import tqdm
import argparse
from utils import print_metrics
from dgl.data.utils import load_graphs
import model as model
import numpy as np
from collections import defaultdict
import dgl

torch.autograd.set_detect_anomaly(True) # For debugging backward

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Career Prediction")
parser.add_argument("--history_length", default=10000, type=int, help="past length for RNNs")
parser.add_argument("--device", default="cuda:0", type=str, help="Which device do you wanna use")
parser.add_argument("--lr", default=0.001, type=float, help="")
parser.add_argument("--emb_dim", default=150, type=int, help="")
parser.add_argument("--batch_size", default=100, type=int, help="")
args = parser.parse_args()


history_length = args.history_length
emb_dim = args.emb_dim
num_epochs = 100000
trainset_batch_size = args.batch_size
test_batch_size = 1000
best_mrr = 0
best_h1 = 0
best_h5 = 0
best_h10 = 0
best_epoch = 0
learning_rate = args.lr
device = args.device
random_seed = 1024
random.seed(random_seed)
torch.manual_seed(random_seed)

with open('NYC_train.pickle', 'rb') as f:
    User_cnt, POI_cnt, Cat_cnt, Time_cnt, POI_dict, cat_dict, time_dict, cat_txt_dict, coordinate_dict, full_time_dict = pickle.load(f)
with open('NYC_test.pickle', 'rb') as f:
    test_user_seed, user_future_POIs, user_future_cats, POI_dict_test, cat_dict_test, time_dict_test, cat_txt_dict_test, coordinate_dict_test, full_time_dict_test = pickle.load(f)
print('Loading graph...')
glist, _ = load_graphs('NYC_train.TKG', [0]) # glist will be [g1]
Train_Graph = glist[0]
splitted_Train_Graph = []
no_edge_graph_for_first = dgl.graph(([], []), num_nodes = User_cnt + POI_cnt)
splitted_Train_Graph.append(no_edge_graph_for_first)
for i in range(Time_cnt):
    graph_at_i = Train_Graph.edge_subgraph(np.where(Train_Graph.edata['time_id'] == i)[0], relabel_nodes = False, store_ids=True)
    splitted_Train_Graph.append(graph_at_i)


user_id_max = User_cnt-1
u, train_POI_list = zip(*list(POI_dict.items()))
u, train_cat_list = zip(*list(cat_dict.items()))
u, train_time_list = zip(*list(time_dict.items()))


train_POI_tensor = []
train_cat_tensor = []
train_time_tensor = []



for p, c, t in zip(train_POI_list, train_cat_list, train_time_list):
    train_POI_tensor.append(torch.tensor(p).type(torch.LongTensor))
    train_cat_tensor.append(torch.tensor(c).type(torch.LongTensor))
    train_time_tensor.append(t)



# test_POI = []
# test_cat = []



test_user_entid2 = []
test_time_batch = []
app_times = defaultdict(list)
filter_poi = defaultdict(list)
no_test_user = list()
for user, poi_list, time_list in zip(u, train_POI_list, train_time_list):
    test_times = time_dict_test[user]
    if len(test_times) == 0:
        no_test_user.append(user)
        continue
    test_user_entid2.append(user)
    test_time_batch.append(test_times[0]) 
    for p, t in zip(poi_list, time_list):
        app_times[u].append(t)
        app_times[p].append(t)
        if t == test_times[0]:
            filter_poi[u].append(p-User_cnt)
print(len(u))
print(len(no_test_user))
print("AA")
label_POI_index = user_future_POIs

for key in app_times.keys():
    app_times[key] = sorted(app_times[key])
remove_list = []
for i in range(Time_cnt):
    remove_list.append(set())
for i in range(User_cnt + POI_cnt):
    try:
        remove_list[app_times[i][0]-1].add(i) 
    except:
        pass
for i in range(Time_cnt):
    remove_list[i] = list(remove_list[i])


label_cat_index = user_future_cats

print("Emb dim:", emb_dim)
print("Batch_size:", trainset_batch_size)
print("learning_rate:", learning_rate)

model = model.GCRNN(User_cnt, POI_cnt, Cat_cnt*2, emb_dim, User_cnt-1, device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
print("Train start")
print(User_cnt)
for epoch in range(num_epochs):
    print("-epoch: ", epoch,"/ 0 ~",num_epochs-1,"processing")
    model.train()
    prev_batch_cnt = 0
    batch_cnt = 0
    epoch_start = time.time()
    for batch_it in tqdm(range(len(train_POI_list) // trainset_batch_size if len(train_POI_list) % trainset_batch_size == 0 else len(train_POI_list) // trainset_batch_size +1)):
        prev_batch_cnt = batch_cnt
        batch_cnt += trainset_batch_size
        if batch_cnt > len(train_POI_list):
            batch_cnt = len(train_POI_list)
        batch_size = batch_cnt - prev_batch_cnt
        POI_loss = model(u[prev_batch_cnt:batch_cnt], train_POI_tensor[prev_batch_cnt:batch_cnt], train_cat_tensor[prev_batch_cnt:batch_cnt], train_time_tensor[prev_batch_cnt:batch_cnt], Train_Graph, splitted_Train_Graph, history_length, remove_list)
        loss = POI_loss
        loss.backward() # calculate gradient
        max_norm_ = 34307752666471.867 * 0.8
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_) # gradient clipping
        optimizer.step() # update parameter via calculated gradient
        optimizer.zero_grad() # initialize
    print("Test start")
    model.eval()
    with torch.no_grad():
        POI_ranks = []
        job_ranks = []
        print(len(test_user_entid2))
        prev_test_batch_cnt = 0
        test_batch_cnt = 0
        for batch_it in tqdm(range(len(test_user_entid2) // test_batch_size if len(test_user_entid2) % test_batch_size == 0 else len(test_user_entid2) // test_batch_size +1)):
            test_batch_cnt+=test_batch_size
            All_UP_score = model.inference(test_user_entid2[prev_test_batch_cnt:test_batch_cnt], test_time_batch[prev_test_batch_cnt:test_batch_cnt], Train_Graph, splitted_Train_Graph, history_length, remove_list)
            for user_id, UP_score in zip(test_user_entid2[prev_test_batch_cnt:test_batch_cnt], All_UP_score):
                user_future_POI_index = list(label_POI_index[user_id])
                POI_label_scores = UP_score[user_future_POI_index]
                for cls_ in POI_label_scores:
                    gap = UP_score - cls_
                    filter_gap = gap[filter_poi[user_id]]
                    POI_ranks.append(len(gap[gap>0]) - len(filter_gap[filter_gap>0]) + 1)
            prev_test_batch_cnt = test_batch_cnt
        mrr, h1, h3, h5, h10 = print_metrics(POI_ranks, job_ranks)
    if mrr > best_mrr:
        best_epoch = epoch
        best_mrr = mrr
        best_h1 = h1
        best_h5 = h5
        best_h10 = h10
    print("Best MRR:", round(best_mrr,4),"H1:", round(best_h1,4),"H5:", round(best_h5,4),"H10:", round(best_h10,4), "at epoch", best_epoch)

