# Vehicle Type Specific Traffic Volume Analysis Using STGCN with LSTM-Based Residual Correction

Lane-level, vehicle-type-specific traffic prediction using an STGCN baseline with LSTM-based residual correction on real-world traffic data.

## Summary

This project predicts lane-level traffic conditions for individual vehicle types using real-world traffic data from Bucheon, South Korea. An STGCN first learns spatial and temporal traffic patterns to produce a base prediction. An LSTM then models the remaining residual error and corrects that prediction. The combined approach supports vehicle-type-specific prediction and reduces error compared with the STGCN baseline.

## Dataset

| Item | Description |
| --- | --- |
| Location | Bucheon City, Gyeonggi-do, South Korea |
| Coverage | 9 intersections / 1,370 lanes |
| Period | 30 days between August and October 2022 |
| Sampling interval | 5 minutes (288 intervals per day) |
| Vehicle types | Passenger Car, Bus, Truck, Motorcycle |
| Features | Queue length, average speed |

Each lane is represented as a graph node. Lane-to-lane physical connections form the adjacency matrix, with self-loops included.

## Input and Output

| | Configuration |
| --- | --- |
| Input window | Previous 12 time steps (1 hour) |
| Input channels | Queue length and average speed for four vehicle types (8), plus time-of-day encoding (1): **9 total** |
| Output horizon | Next 5-minute interval |
| Output channels | Queue length and average speed for four vehicle types: **8 total** |

## Proposed Method

**Baseline:** STGCN<br>
**Proposed model:** STGCN + LSTM-based Residual Correction

```text
Traffic Data
    ↓
  STGCN
    ↓
Base Prediction
    ↓
Past Residual Sequence
    ↓
Average Pooling to 439 Connectivity-Based Clusters
    ↓
Cluster-LSTM Residual Correction
    ↓
Broadcast Correction to Lane Nodes
    ↓
Final Prediction
```

The STGCN is retained as the base predictor; it is not replaced by the LSTM. Residuals from 1,370 lane nodes are average-pooled into 439 connectivity-based clusters. A cluster-specific LSTM predicts the correction, which is broadcast back to lane nodes and added to the STGCN output.

## Code

The public reference implementation retains the paper's baseline and proposed architecture while excluding private traffic data, checkpoints, and exploratory experiment branches.

```text
src/models/stgcn.py                 # STGCN baseline
src/models/residual_stgcn_lstm.py   # STGCN + Cluster-LSTM correction
src/train.py                        # Baseline/proposed training entry point
src/evaluate.py                     # MAE and RMSE evaluation
examples/synthetic_data_demo.py     # Runnable smoke test without private data
```

```bash
pip install -r requirements.txt
python examples/synthetic_data_demo.py
```

Private training arrays are intentionally excluded. Their expected layout is documented in [docs/data_format.md](docs/data_format.md).

## Experimental Setup

The evaluation compares the STGCN baseline with the residual-correction model using MAE and RMSE. Experiments were conducted with PyTorch 2.0 on a single NVIDIA RTX-series GPU. A fixed random seed and deterministic settings were applied to support reproducibility.

| Model | Training time |
| --- | --- |
| STGCN baseline | Approximately 2–3 hours |
| STGCN + LSTM Residual Correction | Approximately 3–4 hours |

## Results

**MAE ↓ 45.6%** &nbsp;&nbsp; **RMSE ↓ 49.7%**

| Metric | STGCN | Proposed Model | Improvement |
| --- | ---: | ---: | --- |
| MAE | 0.2848 | 0.1549 | 45.6% reduction |
| RMSE | 0.8573 | 0.4316 | 49.7% reduction |

## Vehicle-Type Findings

Bus and motorcycle traffic had relatively lower volumes than passenger cars and trucks. Larger relative improvements were observed for these vehicle types in the experiments.

## Key Findings

- Modeled lane-level traffic as a graph with 1,370 nodes.
- Performed vehicle-type-specific prediction for four vehicle categories.
- Combined STGCN with LSTM-based residual correction.
- Reduced overall MAE by 45.6% and RMSE by 49.7%.
- Observed larger relative improvements for buses and motorcycles.

## My Role

- First author
- Model implementation
- Model experimentation
- Paper writing


## Tech Stack & Skills

### Modeling approaches

- PyTorch 2.0
- Spatio-temporal graph convolutional networks (STGCN)
- LSTM-based time-series residual modeling
- Graph-based traffic modeling
- Lane-level, vehicle-type-specific traffic prediction
- MAE and RMSE model evaluation

## Paper / Conference

**Paper:** *Vehicle Type Specific Traffic Volume Analysis Using STGCN with LSTM-Based Residual Correction*<br>
**Authors:** Seongon Moon, Jongseok Min, Byeongyoon An, Junyeong Lim, and Jinho Lee<br>
**Conference:** IEEE International Conference on Industrial Informatics (INDIN 2026)<br>
**Status:** Final manuscript submitted; presented at the conference.
