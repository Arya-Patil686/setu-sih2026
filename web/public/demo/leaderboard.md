# SETU evaluation - controlled benchmark with exact ground truth

| Setting | Value |
|---|---|
| pairs | 31 |
| methods | 12 |
| azimuth sweep | delta sun azimuth 0 to 180 deg at 25 deg elevation |
| elevation sweep | source sun elevation 10 to 75 deg against a 45 deg reference |
| scale sweep | GSD ratio 1x to 16x |
| terrain | highland, mare |
| ground truth | exact - both images rendered from one DEM under a known warp |

### All methods

| Method | Tie-point RMSE (px) | Model RMSE vs truth (px) | Precision @3px | Inlier ratio | Inliers | Coverage @150 pts | Clark-Evans R @150 | Time (s) |
|---|---|---|---|---|---|---|---|---|
| setu_no_refine <sub>(4/31 failed)</sub> | 0.584<br><sub>[0.317, 1.085]</sub> | 0.818<br><sub>[0.073, 2.298]</sub> | 99.4%<br><sub>[98.1%, 100.0%]</sub> | 96.3%<br><sub>[88.9%, 100.0%]</sub> | 222<br><sub>[190, 251]</sub> | 0.78<br><sub>[0.70, 0.84]</sub> | 0.89<br><sub>[0.84, 0.92]</sub> | 4.2<br><sub>[3.8, 4.7]</sub> |
| setu_no_gate <sub>(4/31 failed)</sub> | 0.667<br><sub>[0.379, 1.140]</sub> | 0.894<br><sub>[0.087, 2.055]</sub> | 98.8%<br><sub>[96.9%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 177<br><sub>[150, 201]</sub> | 0.72<br><sub>[0.64, 0.78]</sub> | 0.81<br><sub>[0.74, 0.85]</sub> | 5.1<br><sub>[4.7, 5.6]</sub> |
| setu_no_uniform <sub>(4/31 failed)</sub> | 0.463<br><sub>[0.365, 0.606]</sub> | 0.897<br><sub>[0.084, 2.383]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 690<br><sub>[544, 833]</sub> | 0.59<br><sub>[0.50, 0.66]</sub> | 0.72<br><sub>[0.67, 0.76]</sub> | 4.4<br><sub>[4.0, 4.7]</sub> |
| **setu_full** <sub>(4/31 failed)</sub> | 0.477<br><sub>[0.374, 0.618]</sub> | 0.897<br><sub>[0.084, 2.383]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 209<br><sub>[177, 238]</sub> | 0.74<br><sub>[0.66, 0.81]</sub> | 0.88<br><sub>[0.81, 0.92]</sub> | 6.3<br><sub>[5.8, 6.8]</sub> |
| setu_no_reillum <sub>(9/31 failed)</sub> | 3.614<br><sub>[0.570, 9.585]</sub> | 5.744<br><sub>[0.243, 16.552]</sub> | 93.2%<br><sub>[84.1%, 100.0%]</sub> | 81.8%<br><sub>[63.6%, 95.5%]</sub> | 85<br><sub>[54, 118]</sub> | 0.61<br><sub>[0.51, 0.71]</sub> | 0.87<br><sub>[0.81, 0.92]</sub> | 7.2<br><sub>[5.7, 8.6]</sub> |
| loftr | 54.457<br><sub>[3.354, 148.363]</sub> | 69.455<br><sub>[4.570, 184.094]</sub> | 61.2%<br><sub>[46.1%, 76.9%]</sub> | 63.1%<br><sub>[48.6%, 78.0%]</sub> | 1527<br><sub>[1062, 2009]</sub> | 0.58<br><sub>[0.46, 0.71]</sub> | 0.79<br><sub>[0.69, 0.88]</sub> | 5.5<br><sub>[0.8, 13.3]</sub> |
| intfeat | 66.503<br><sub>[30.732, 103.318]</sub> | 106.574<br><sub>[52.082, 164.149]</sub> | 39.7%<br><sub>[28.0%, 52.2%]</sub> | 46.6%<br><sub>[37.3%, 56.4%]</sub> | 369<br><sub>[161, 587]</sub> | 0.33<br><sub>[0.24, 0.43]</sub> | 0.52<br><sub>[0.41, 0.64]</sub> | 0.2<br><sub>[0.1, 0.3]</sub> |
| cfog | 96.567<br><sub>[7.178, 231.803]</sub> | 112.331<br><sub>[9.169, 269.750]</sub> | 55.7%<br><sub>[43.6%, 66.3%]</sub> | 62.1%<br><sub>[53.0%, 70.8%]</sub> | 159<br><sub>[136, 181]</sub> | 0.63<br><sub>[0.53, 0.71]</sub> | 1.10<br><sub>[0.98, 1.20]</sub> | 0.4<br><sub>[0.4, 0.5]</sub> |
| rift | 60.276<br><sub>[26.906, 97.363]</sub> | 156.087<br><sub>[44.541, 316.878]</sub> | 55.8%<br><sub>[40.1%, 72.1%]</sub> | 63.5%<br><sub>[51.3%, 74.6%]</sub> | 91<br><sub>[57, 126]</sub> | 0.35<br><sub>[0.25, 0.44]</sub> | 0.70<br><sub>[0.61, 0.79]</sub> | 0.8<br><sub>[0.6, 1.2]</sub> |
| sift | 68.422<br><sub>[33.419, 106.569]</sub> | 184.232<br><sub>[76.373, 312.959]</sub> | 50.3%<br><sub>[36.5%, 65.1%]</sub> | 62.2%<br><sub>[52.5%, 72.8%]</sub> | 173<br><sub>[73, 271]</sub> | 0.33<br><sub>[0.23, 0.44]</sub> | 0.77<br><sub>[0.63, 0.94]</sub> | 0.1<br><sub>[0.0, 0.1]</sub> |
| orb | 99.880<br><sub>[28.692, 202.718]</sub> | 327.076<br><sub>[74.472, 760.874]</sub> | 46.4%<br><sub>[32.4%, 60.8%]</sub> | 54.4%<br><sub>[42.8%, 66.2%]</sub> | 343<br><sub>[146, 543]</sub> | 0.25<br><sub>[0.17, 0.33]</sub> | 0.37<br><sub>[0.29, 0.45]</sub> | 0.0<br><sub>[0.0, 0.0]</sub> |
| disk_lightglue | 100.382<br><sub>[33.916, 181.208]</sub> | 446.074<br><sub>[83.806, 1068.254]</sub> | 60.8%<br><sub>[45.6%, 77.1%]</sub> | 73.4%<br><sub>[62.3%, 83.8%]</sub> | 823<br><sub>[592, 1064]</sub> | 0.44<br><sub>[0.34, 0.55]</sub> | 0.73<br><sub>[0.64, 0.82]</sub> | 1.3<br><sub>[0.4, 2.9]</sub> |

**Tie-point RMSE** is the error of the delivered correspondences against exact truth, over inliers, and it is the number the problem statement's sub-pixel requirement refers to. **Model RMSE** is the error of the fitted transform; it falls roughly as the square root of the point count, so a method returning thousands of noisy points can score well on it while its individual tie points are useless. Uniformity is measured after subsampling every method to the same 150 points, because coverage of an 8x8 lattice is otherwise a measure of density rather than of distribution. Square brackets are bootstrap 95% confidence intervals over pairs.

### Ablation table

| Configuration | Tie-point RMSE (px) | Model RMSE (px) | Precision @3px | Inlier ratio | Coverage |
|---|---|---|---|---|---|
| SETU without sun-synchronised re-illumination (ablates N1) | 3.614 | 5.744 | 93.2% | 81.8% | 0.63 |
| SETU without the agreement gate, track A only (ablates N2) | 0.667 | 0.894 | 98.8% | 100.0% | 0.76 |
| SETU without sub-pixel refinement (ablates N4) | 0.584 | 0.818 | 99.4% | 96.3% | 0.83 |
| SETU without uniformity enforcement (ablates N3) | 0.463 | 0.897 | 100.0% | 100.0% | 0.70 |
| SETU, complete | 0.477 | 0.897 | 100.0% | 100.0% | 0.78 |

- **SETU without sun-synchronised re-illumination (ablates N1)**: tie-point RMSE 3.614 px against 0.477 px, a factor of 7.6.

- **SETU without the agreement gate, track A only (ablates N2)**: tie-point RMSE 0.667 px against 0.477 px, a factor of 1.4.

- **SETU without sub-pixel refinement (ablates N4)**: tie-point RMSE 0.584 px against 0.477 px, a factor of 1.2.

- **SETU without uniformity enforcement (ablates N3)**: tie-point RMSE 0.463 px against 0.477 px, a factor of 1.0.