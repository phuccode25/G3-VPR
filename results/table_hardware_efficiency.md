| Method | Net Params (M) | Hidden PCA (M) | Total Stored (M) | Desc Dim | 1M-Map RAM (GB) | BS=1 Latency (ms) | Throughput (img/s) | BS=1 Invariant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CricaVPR (Teacher)** | 106.76 | 0.00 | 106.76 | 10752 | 40.05 | 53.00 | 44.0 | No (Drops -24%) |
| **D2VPR (Encoder, Raw)** | 38.24 | 0.00 | 38.24 | 10752 | 40.05 | 58.15 | 101.3 | No (Drops -24%) |
| **D2VPR (Encoder + PCA4096)** | 38.24 | 44.04 | 82.28 | 4096 | 15.26 | 77.05 | 100.8 | No (Drops -24%) |
| **D2VPR (No Encoder)** | 27.21 | 0.00 | 27.21 | 10752 | 40.05 | 61.58 | 101.9 | Yes (Exact) |
| **G3-VPR (Ours, 4096-D Max)** | 30.97 | 0.00 | 30.97 | 4096 | 15.26 | 57.62 | 125.7 | Yes (Exact) |
| **G3-VPR (Ours, 512-D Balanced)** | 30.97 | 0.00 | 30.97 | 512 | 1.91 | 57.62 | 125.7 | Yes (Exact) |
| **G3-VPR (Ours, 256-D Edge)** | 30.97 | 0.00 | 30.97 | 256 | 0.95 | 57.62 | 125.7 | Yes (Exact) |
| **G3-VPR (Ours, 128-D Ultra-Compact)** | 30.97 | 0.00 | 30.97 | 128 | 0.48 | 57.62 | 125.7 | Yes (Exact) |
