# Ablations and architectural alternatives

Every variant now runs from the single implementation in `../INDIANA/`, selected with `--variant`.
The previous `EQ2/{KG,TG}` and `EQ3/{ATT,GAT,GRU}` directories were removed: they were full copies of
`INDIANA/model.py`, and `EQ3/GAT` was byte-identical to it, so the GAT column of Table 5 was produced
by INDIANA's own code rather than by GAT.

| Table | Row | Command |
|-------|-----|---------|
| 4 (a) | Baseline  | `--variant baseline` |
| 4 (b) | KG        | `--variant kg` |
| 4 (c) | TG        | `--variant tg` |
| 4     | I-TKG     | `--variant indiana` |
| 5     | GAT       | `--variant gat` |
| 5     | ATT       | `--variant att` |
| 5     | GRU       | `--variant gru` |
| 5     | KGNN+GRNN | `--variant indiana` |

```
python main.py --dataset NYC --data_dir ../../data --device cuda:0 --variant gat
```

Table 4 (d) `Unified` and (e) `Node-Cat` are **not implemented**. They change how the graph is built,
not how the model runs, and no code for either exists in this repository.
