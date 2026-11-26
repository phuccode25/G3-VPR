| Method | Net Params (M) | Hidden PCA (M) | Total Stored (M) | Desc Dim | 1M-Map RAM (GB) | BS=1 Latency (ms) | Throughput (img/s) | BS=1 Invariant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CricaVPR (Teacher)** | 106.76 | 0.00 | 106.76 | 10752 | 40.05 | 36.18 | 43.7 | No (Drops -24%) |
| **D2VPR (Encoder, Raw)** | 38.24 | 0.00 | 38.24 | 10752 | 40.05 | 38.70 | 100.5 | No (Drops -24%) |
| **D2VPR (Encoder + PCA4096)** | 38.24 | 44.04 | 82.28 | 4096 | 15.26 | 37.45 | 99.9 | No (Drops -24%) |
| **D2VPR (No Encoder)** | 27.21 | 0.00 | 27.21 | 10752 | 40.05 | 37.72 | 101.0 | Yes (Exact) |
| **G3-VPR (Ours, 4096-D Max)** | 36.01 | 0.00 | 36.01 | 4096 | 15.26 | 30.95 | 123.9 | Yes (Exact) |
| **G3-VPR (Ours, 512-D Balanced)** | 36.01 | 0.00 | 36.01 | 512 | 1.91 | 30.95 | 123.9 | Yes (Exact) |
| **G3-VPR (Ours, 256-D Edge)** | 36.01 | 0.00 | 36.01 | 256 | 0.95 | 30.95 | 123.9 | Yes (Exact) |
| **G3-VPR (Ours, 128-D Ultra-Compact)** | 36.01 | 0.00 | 36.01 | 128 | 0.48 | 30.95 | 123.9 | Yes (Exact) |
