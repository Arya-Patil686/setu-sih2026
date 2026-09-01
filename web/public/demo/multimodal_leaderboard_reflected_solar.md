# IIRS pseudo-panchromatic (900-1600 nm) against a visible reference

| Setting | Value |
|---|---|
| pairs | 10 |
| scale ratio | 4x (80 m IIRS against ~20 m Kaguya TC) |
| sun azimuth difference | 0 to 180 deg at 25 deg elevation |
| sensor gap | IIRS SNR 35 with column striping, against a framing camera at SNR 95 |

### All methods

| Method | Tie-point RMSE (px) | Model RMSE vs truth (px) | Precision @3px | Inlier ratio | Inliers | Coverage @150 pts | Clark-Evans R @150 | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **setu_full** | 2.809<br><sub>[2.477, 3.194]</sub> | 2.725<br><sub>[2.157, 3.415]</sub> | 64.5%<br><sub>[54.7%, 72.6%]</sub> | 87.5%<br><sub>[80.9%, 92.9%]</sub> | 53<br><sub>[37, 68]</sub> | 0.34<br><sub>[0.26, 0.43]</sub> | 0.62<br><sub>[0.53, 0.72]</sub> | 1.5<br><sub>[1.3, 1.6]</sub> |
| setu_no_reillum <sub>(2/10 failed)</sub> | 2.819<br><sub>[2.492, 3.201]</sub> | 5.757<br><sub>[2.456, 10.706]</sub> | 58.2%<br><sub>[37.7%, 76.7%]</sub> | 65.2%<br><sub>[36.7%, 90.2%]</sub> | 28<br><sub>[9, 50]</sub> | 0.23<br><sub>[0.10, 0.38]</sub> | 0.49<br><sub>[0.35, 0.65]</sub> | 0.8<br><sub>[0.7, 1.0]</sub> |
| loftr | 310.894<br><sub>[168.512, 469.205]</sub> | 362.460<br><sub>[209.052, 513.250]</sub> | 0.9%<br><sub>[0.0%, 1.9%]</sub> | 6.0%<br><sub>[5.3%, 6.8%]</sub> | 6<br><sub>[5, 8]</sub> | 0.08<br><sub>[0.06, 0.10]</sub> | 0.72<br><sub>[0.53, 0.90]</sub> | 0.6<br><sub>[0.6, 0.7]</sub> |
| orb | 351.790<br><sub>[229.979, 453.116]</sub> | 561.609<br><sub>[368.007, 762.256]</sub> | 6.4%<br><sub>[0.0%, 14.9%]</sub> | 25.1%<br><sub>[18.5%, 32.4%]</sub> | 10<br><sub>[4, 22]</sub> | 0.08<br><sub>[0.04, 0.15]</sub> | 0.59<br><sub>[0.36, 0.87]</sub> | 0.0<br><sub>[0.0, 0.0]</sub> |
| cfog | 564.696<br><sub>[531.225, 594.927]</sub> | 621.616<br><sub>[612.044, 630.818]</sub> | 0.0%<br><sub>[0.0%, 0.0%]</sub> | 11.0%<br><sub>[10.0%, 12.0%]</sub> | 28<br><sub>[26, 31]</sub> | 0.15<br><sub>[0.14, 0.17]</sub> | 0.42<br><sub>[0.39, 0.46]</sub> | 0.6<br><sub>[0.6, 0.7]</sub> |
| rift | 441.393<br><sub>[341.698, 538.929]</sub> | 970.466<br><sub>[611.425, 1420.333]</sub> | 0.0%<br><sub>[0.0%, 0.0%]</sub> | 35.2%<br><sub>[20.8%, 50.2%]</sub> | 3<br><sub>[2, 4]</sub> | 0.04<br><sub>[0.03, 0.05]</sub> | 0.53<br><sub>[0.39, 0.66]</sub> | 1.1<br><sub>[1.1, 1.1]</sub> |
| sift | 447.252<br><sub>[271.829, 619.092]</sub> | 1166.300<br><sub>[488.301, 2062.406]</sub> | 12.7%<br><sub>[0.0%, 30.2%]</sub> | 48.8%<br><sub>[41.3%, 58.0%]</sub> | 27<br><sub>[3, 73]</sub> | 0.13<br><sub>[0.05, 0.27]</sub> | 0.88<br><sub>[0.71, 1.05]</sub> | 0.1<br><sub>[0.1, 0.1]</sub> |

**Tie-point RMSE** is the error of the delivered correspondences against exact truth, over inliers, and it is the number the problem statement's sub-pixel requirement refers to. **Model RMSE** is the error of the fitted transform; it falls roughly as the square root of the point count, so a method returning thousands of noisy points can score well on it while its individual tie points are useless. Uniformity is measured after subsampling every method to the same 150 points, because coverage of an 8x8 lattice is otherwise a measure of density rather than of distribution. Square brackets are bootstrap 95% confidence intervals over pairs.

### Ablation table

| Configuration | Tie-point RMSE (px) | Model RMSE (px) | Precision @3px | Inlier ratio | Coverage |
|---|---|---|---|---|---|
| SETU without sun-synchronised re-illumination (ablates N1) | 2.819 | 5.757 | 58.2% | 65.2% | 0.23 |
| SETU, complete | 2.809 | 2.725 | 64.5% | 87.5% | 0.34 |

- **SETU without sun-synchronised re-illumination (ablates N1)**: tie-point RMSE 2.819 px against 2.809 px, a factor of 1.0.