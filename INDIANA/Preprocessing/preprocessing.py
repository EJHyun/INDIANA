import argparse
import csv
import os
import pickle
from collections import defaultdict

import dgl
import numpy as np
import torch
from dgl.data.utils import save_graphs

MONTHS = {m: '{:02d}'.format(i + 1) for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}


def read_foursquare(path):
    records = []
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            if not line.strip():
                continue
            field = line.rstrip('\n').split('\t')
            raw = field[7].replace('+0000', '').replace('  ', ' ').strip()
            part = raw.split(' ')
            slot = '{} {} {} {}'.format(part[4], MONTHS[part[1]], part[2], part[3][:2])
            records.append({'user': field[0], 'poi': field[1], 'cat': field[2],
                            'cat_txt': field[3], 'lat': field[4], 'lon': field[5],
                            'slot': slot, 'full_time': raw})
    return records


def read_gowalla(paths):
    records = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                stamp = row['checkin_time']
                slot = '{} {} {} {}'.format(stamp[0:4], stamp[5:7], stamp[8:10], stamp[11:13])
                records.append({'user': row['user_id'], 'poi': row['POI_id'], 'cat': row['POI_catid_code'],
                                'cat_txt': row['POI_catname'], 'lat': row['latitude'], 'lon': row['longitude'],
                                'slot': slot, 'full_time': stamp})
    return records


def build(records, out_dir, dataset, min_poi_checkins, test_ratio, test_slots):
    poi_checkins = defaultdict(int)
    for r in records:
        poi_checkins[r['poi']] += 1
    records = [r for r in records if poi_checkins[r['poi']] >= min_poi_checkins]

    usr_list = sorted({r['user'] for r in records})
    poi_list = sorted({r['poi'] for r in records})
    cat_list = sorted({r['cat'] for r in records})
    slot_list = sorted({r['slot'] for r in records})
    usr_id = {v: i for i, v in enumerate(usr_list)}
    poi_id = {v: i for i, v in enumerate(poi_list)}
    cat_id = {v: i for i, v in enumerate(cat_list)}
    slot_id = {v: i for i, v in enumerate(slot_list)}

    User_cnt, POI_cnt, Cat_cnt, Time_cnt = len(usr_list), len(poi_list), len(cat_list), len(slot_list)
    cut = Time_cnt - (test_slots if test_slots > 0 else int(round(Time_cnt * test_ratio)))

    print('=== {} ==='.format(dataset))
    print('|U| {}  |C| {}  |T| {}  |P| {}  #(Check-ins) {}'.format(
        User_cnt, Cat_cnt, Time_cnt, POI_cnt, len(records)))
    print('train time-windows [0, {}), test time-windows [{}, {})'.format(cut, cut, Time_cnt))

    train = [defaultdict(list) for _ in range(6)]
    test = [defaultdict(list) for _ in range(6)]
    test_user_seed = set()
    user_future_POIs = defaultdict(set)
    user_future_cats = defaultdict(set)
    src_ids, dst_ids, cat_ids, time_ids = [], [], [], []
    all_src, all_dst, all_cat, all_time = [], [], [], []
    n_train, n_test = 0, 0

    for r in records:
        u, p, c, t = usr_id[r['user']], poi_id[r['poi']], cat_id[r['cat']], slot_id[r['slot']]
        all_src.extend([u, User_cnt + p])
        all_dst.extend([User_cnt + p, u])
        all_cat.extend([c, Cat_cnt + c])
        all_time.extend([t, t])
        if t < cut:
            for d, v in zip(train, [User_cnt + p, c, t, r['cat_txt'], (r['lat'], r['lon']), r['full_time']]):
                d[u].append(v)
            src_ids.append(u)
            dst_ids.append(User_cnt + p)
            cat_ids.append(c)
            time_ids.append(t)
            src_ids.append(User_cnt + p)
            dst_ids.append(u)
            cat_ids.append(Cat_cnt + c)
            time_ids.append(t)
            n_train += 1
        else:
            for d, v in zip(test, [p, c, t, r['cat_txt'], (r['lat'], r['lon']), r['full_time']]):
                d[u].append(v)
            user_future_POIs[u].add(p)
            user_future_cats[u].add(c)
            n_test += 1

    for u in list(train[0].keys()):
        if len(test[2][u]) > 0:
            test_user_seed.add(u)

    print('train check-ins {} ({:.2f}%)  test check-ins {} ({:.2f}%)  test users {}'.format(
        n_train, 100.0 * n_train / len(records), n_test, 100.0 * n_test / len(records), len(test_user_seed)))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, dataset + '_train.pickle'), 'wb') as fw:
        pickle.dump([User_cnt, POI_cnt, Cat_cnt, Time_cnt] + train, fw)
    with open(os.path.join(out_dir, dataset + '_test.pickle'), 'wb') as fw:
        pickle.dump([test_user_seed, user_future_POIs, user_future_cats] + test, fw)

    g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)), num_nodes=User_cnt + POI_cnt)
    g.ndata['ent_id'] = torch.tensor(list(range(User_cnt + POI_cnt)))
    g.edata['cat_id'] = torch.tensor(cat_ids)
    g.edata['time_id'] = torch.tensor(time_ids)
    assert int(g.edata['time_id'].max()) < cut
    save_graphs(os.path.join(out_dir, dataset + '_train.TKG'), [g], {dataset: torch.tensor([0])})
    print(g)

    ga = dgl.graph((torch.tensor(all_src), torch.tensor(all_dst)), num_nodes=User_cnt + POI_cnt)
    ga.ndata['ent_id'] = torch.tensor(list(range(User_cnt + POI_cnt)))
    ga.edata['cat_id'] = torch.tensor(all_cat)
    ga.edata['time_id'] = torch.tensor(all_time)
    save_graphs(os.path.join(out_dir, dataset + '_all.TKG'), [ga], {dataset: torch.tensor([0])})
    print('eval graph edges', ga.number_of_edges())


def main():
    parser = argparse.ArgumentParser(description="Build the I-TKG inputs for INDIANA")
    parser.add_argument('--dataset', required=True, choices=['NYC', 'TKY', 'CA'])
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--min_poi_checkins', default=10, type=int)
    parser.add_argument('--test_ratio', default=0.1, type=float)
    parser.add_argument('--test_slots', default=0, type=int, help='absolute number of trailing time-windows to hold out, overrides --test_ratio')
    args = parser.parse_args()

    if args.dataset == 'CA':
        records = read_gowalla([os.path.join(args.raw_dir, 'Gowalla-CA_{}.csv'.format(s))
                                for s in ['train', 'val', 'test']])
    else:
        records = read_foursquare(os.path.join(
            args.raw_dir, 'dataset_TSMC2014_{}.txt'.format(args.dataset)))
    print('raw check-ins', len(records))
    build(records, args.out_dir, args.dataset, args.min_poi_checkins, args.test_ratio, args.test_slots)


if __name__ == '__main__':
    main()
