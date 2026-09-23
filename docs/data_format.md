# Private Data Format

The original traffic data is not included in this repository. Training requires
private, preprocessed arrays with the following layout.

| File | Required arrays / shape |
| --- | --- |
| `windows.npz` | `x`: `(M, 9, 12, N)`, `y`: `(M, 8, N)`, `residuals`: `(M, 12, 8, N)` |
| `adjacency.npy` | Lane adjacency matrix with self-loops: `(N, N)` |
| `cluster_ids.npy` | Connectivity-cluster ID per lane: `(N,)` |

`residuals` are the past residual sequences used by the correction module. For
the study, `N=1,370` and the lane connectivity partition has `K=439` clusters.
