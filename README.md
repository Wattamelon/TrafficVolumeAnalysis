# Vehicle Type Specific Traffic Volume Analysis Using STGCN with LSTM-Based Residual Correction

Lane-level, vehicle-type-specific traffic prediction using an STGCN baseline with LSTM-based residual correction on real-world traffic data.

## Summary

This project predicts lane-level traffic conditions for individual vehicle types using real-world traffic data from Bucheon, South Korea. An STGCN first learns spatial and temporal traffic patterns to produce a base prediction. An LSTM then models the remaining residual error and corrects that prediction. The combined approach supports vehicle-type-specific prediction and reduces error compared with the STGCN baseline.

## Research Motivation

Traffic prediction often treats all vehicles as a single traffic flow, although passenger cars, buses, trucks, and motorcycles can exhibit different queue-length and average-speed patterns. In addition, temporal patterns can remain in the residual error after an STGCN prediction. This work models lane-level traffic by vehicle type and uses LSTM-based residual correction to refine the STGCN output.

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
Residual Error Modeling
    ↓
LSTM-based Residual Correction
    ↓
Final Prediction
```

The STGCN is retained as the base predictor; it is not replaced by the LSTM. The LSTM predicts residual errors left by the base prediction, which are added back to form the final result.

## Experimental Setup

The evaluation compares the STGCN baseline with the residual-correction model using MAE and RMSE.

| Model | Approximate parameters |
| --- | ---: |
| STGCN | 1.5M |
| STGCN + LSTM Residual Correction | 1.8M |

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

## Tech Stack & Skills

### Modeling approaches

- Spatio-temporal graph convolutional networks (STGCN)
- LSTM-based time-series residual modeling
- Graph-based traffic modeling
- Lane-level, vehicle-type-specific traffic prediction
- MAE and RMSE model evaluation

Implementation libraries are not listed because the current repository does not include source or dependency files from which to verify them.

## Paper / Conference

**Paper:** *Vehicle Type Specific Traffic Volume Analysis Using STGCN with LSTM-Based Residual Correction*<br>
**Conference:** IEEE International Conference on Industrial Informatics (INDIN 2026)

## Repository Notes

This repository currently provides the project overview. No paper PDF, implementation source, dependency manifest, or figures are included, so no publication link, code-specific library list, or visual assets are claimed here.
