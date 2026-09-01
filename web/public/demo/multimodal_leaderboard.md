# Multi-modal sweep - thermal-like IIRS source against a visible reference

| Setting | Value |
|---|---|
| pairs | 10 |
| modality | thermal-like source vs visible reference |
| scale ratio | 4x (80 m IIRS against ~20 m Kaguya TC) |
| sun azimuth difference | 0 to 180 deg at 25 deg elevation |

### All methods

| Method | Tie-point RMSE (px) | Model RMSE vs truth (px) | Precision @3px | Inlier ratio | Inliers | Coverage @150 pts | Clark-Evans R @150 | Time (s) |
|---|---|---|---|---|---|---|---|---|
| cfog | 593.510<br><sub>[564.471, 619.197]</sub> | 622.298<br><sub>[610.518, 634.204]</sub> | 0.0%<br><sub>[0.0%, 0.0%]</sub> | 10.2%<br><sub>[9.1%, 11.2%]</sub> | 26<br><sub>[23, 29]</sub> | 0.15<br><sub>[0.13, 0.17]</sub> | 0.39<br><sub>[0.36, 0.42]</sub> | 0.7<br><sub>[0.7, 0.8]</sub> |
| loftr | 427.922<br><sub>[374.295, 480.878]</sub> | 638.176<br><sub>[457.867, 888.038]</sub> | 0.0%<br><sub>[0.0%, 0.0%]</sub> | 8.2%<br><sub>[7.1%, 9.4%]</sub> | 4<br><sub>[4, 4]</sub> | 0.05<br><sub>[0.05, 0.06]</sub> | 0.85<br><sub>[0.62, 1.09]</sub> | 0.7<br><sub>[0.6, 0.8]</sub> |
| rift | 495.730<br><sub>[366.796, 572.391]</sub> | 665.435<br><sub>[552.762, 845.137]</sub> | 0.0%<br><sub>[0.0%, 0.0%]</sub> | 10.8%<br><sub>[0.0%, 24.6%]</sub> | 2<br><sub>[0, 4]</sub> | 0.07<br><sub>[0.03, 0.16]</sub> | 0.33<br><sub>[0.06, 0.59]</sub> | 1.2<br><sub>[1.1, 1.2]</sub> |
| sift | 450.499<br><sub>[408.356, 492.641]</sub> | 833.304<br><sub>[829.990, 836.618]</sub> | 0.0%<br><sub>[0.0%, 0.0%]</sub> | 12.0%<br><sub>[0.0%, 30.0%]</sub> | 1<br><sub>[0, 2]</sub> | 0.05<br><sub>[0.05, 0.05]</sub> | 1.08<br><sub>[0.62, 1.54]</sub> | 0.1<br><sub>[0.1, 0.1]</sub> |
| orb | 455.163<br><sub>[364.666, 552.506]</sub> | 842.044<br><sub>[522.366, 1386.570]</sub> | 0.3%<br><sub>[0.0%, 1.0%]</sub> | 18.2%<br><sub>[14.9%, 22.0%]</sub> | 6<br><sub>[4, 7]</sub> | 0.05<br><sub>[0.04, 0.05]</sub> | 0.43<br><sub>[0.21, 0.65]</sub> | 0.0<br><sub>[0.0, 0.1]</sub> |
| setu_no_reillum <sub>(10/10 failed)</sub> | no registration | no registration | no registration | no registration | no registration | no registration | no registration | no registration |
| **setu_full** <sub>(10/10 failed)</sub> | no registration | no registration | no registration | no registration | no registration | no registration | no registration | no registration |

**Tie-point RMSE** is the error of the delivered correspondences against exact truth, over inliers, and it is the number the problem statement's sub-pixel requirement refers to. **Model RMSE** is the error of the fitted transform; it falls roughly as the square root of the point count, so a method returning thousands of noisy points can score well on it while its individual tie points are useless. Uniformity is measured after subsampling every method to the same 150 points, because coverage of an 8x8 lattice is otherwise a measure of density rather than of distribution. Square brackets are bootstrap 95% confidence intervals over pairs.

### Ablation table

| Configuration | Tie-point RMSE (px) | Model RMSE (px) | Precision @3px | Inlier ratio | Coverage |
|---|---|---|---|---|---|
| SETU without sun-synchronised re-illumination (ablates N1) | no registration | no registration | no registration | no registration | no registration |
| SETU, complete | no registration | no registration | no registration | no registration | no registration |