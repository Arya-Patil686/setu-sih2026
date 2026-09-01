### Elevation sweep (source 10-75 deg against a 45 deg reference)

| Method | Tie-point RMSE (px) | Model RMSE vs truth (px) | Precision @3px | Inlier ratio | Inliers | Coverage @150 pts | Clark-Evans R @150 | Time (s) |
|---|---|---|---|---|---|---|---|---|
| disk_lightglue | 1.224<br><sub>[1.099, 1.362]</sub> | 0.094<br><sub>[0.075, 0.119]</sub> | 96.9%<br><sub>[94.8%, 98.4%]</sub> | 96.8%<br><sub>[94.8%, 98.3%]</sub> | 1360<br><sub>[1235, 1469]</sub> | 0.65<br><sub>[0.62, 0.68]</sub> | 0.79<br><sub>[0.77, 0.81]</sub> | 0.4<br><sub>[0.3, 0.5]</sub> |
| loftr | 0.712<br><sub>[0.567, 0.892]</sub> | 0.137<br><sub>[0.115, 0.160]</sub> | 98.4%<br><sub>[96.6%, 99.6%]</sub> | 98.4%<br><sub>[96.6%, 99.5%]</sub> | 2807<br><sub>[2575, 3014]</sub> | 0.84<br><sub>[0.81, 0.87]</sub> | 0.93<br><sub>[0.91, 0.96]</sub> | 1.3<br><sub>[1.1, 1.5]</sub> |
| sift | 0.640<br><sub>[0.504, 0.803]</sub> | 0.249<br><sub>[0.129, 0.396]</sub> | 81.9%<br><sub>[68.8%, 92.9%]</sub> | 81.9%<br><sub>[68.8%, 92.8%]</sub> | 326<br><sub>[152, 507]</sub> | 0.51<br><sub>[0.34, 0.68]</sub> | 0.67<br><sub>[0.52, 0.80]</sub> | 0.0<br><sub>[0.0, 0.1]</sub> |
| orb | 1.330<br><sub>[1.255, 1.423]</sub> | 0.337<br><sub>[0.175, 0.520]</sub> | 83.7%<br><sub>[75.7%, 90.3%]</sub> | 84.1%<br><sub>[76.4%, 90.5%]</sub> | 656<br><sub>[336, 980]</sub> | 0.41<br><sub>[0.29, 0.52]</sub> | 0.37<br><sub>[0.23, 0.50]</sub> | 0.0<br><sub>[0.0, 0.0]</sub> |
| rift | 0.859<br><sub>[0.658, 1.089]</sub> | 0.357<br><sub>[0.219, 0.530]</sub> | 86.7%<br><sub>[75.9%, 95.1%]</sub> | 86.8%<br><sub>[76.2%, 95.1%]</sub> | 152<br><sub>[103, 204]</sub> | 0.54<br><sub>[0.46, 0.60]</sub> | 0.64<br><sub>[0.57, 0.70]</sub> | 0.7<br><sub>[0.6, 0.7]</sub> |
| intfeat | 1.119<br><sub>[1.029, 1.213]</sub> | 0.555<br><sub>[0.301, 0.886]</sub> | 63.7%<br><sub>[54.9%, 72.6%]</sub> | 63.4%<br><sub>[54.3%, 72.5%]</sub> | 693<br><sub>[337, 1060]</sub> | 0.50<br><sub>[0.36, 0.65]</sub> | 0.48<br><sub>[0.33, 0.63]</sub> | 0.2<br><sub>[0.1, 0.2]</sub> |
| cfog | 1.356<br><sub>[1.184, 1.518]</sub> | 1.077<br><sub>[0.876, 1.290]</sub> | 74.0%<br><sub>[63.3%, 84.2%]</sub> | 75.8%<br><sub>[65.0%, 86.0%]</sub> | 194<br><sub>[166, 220]</sub> | 0.75<br><sub>[0.69, 0.82]</sub> | 1.26<br><sub>[1.22, 1.28]</sub> | 0.4<br><sub>[0.4, 0.4]</sub> |
| setu_no_refine | 0.961<br><sub>[0.345, 2.061]</sub> | 1.839<br><sub>[0.084, 5.177]</sub> | 98.6%<br><sub>[95.8%, 100.0%]</sub> | 91.7%<br><sub>[75.0%, 100.0%]</sub> | 180<br><sub>[116, 236]</sub> | 0.73<br><sub>[0.57, 0.85]</sub> | 0.87<br><sub>[0.79, 0.94]</sub> | 4.8<br><sub>[3.9, 5.9]</sub> |
| setu_no_gate | 1.046<br><sub>[0.415, 2.093]</sub> | 1.938<br><sub>[0.156, 4.446]</sub> | 97.2%<br><sub>[93.1%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 139<br><sub>[91, 183]</sub> | 0.64<br><sub>[0.48, 0.77]</sub> | 0.82<br><sub>[0.69, 0.91]</sub> | 5.7<br><sub>[4.9, 6.6]</sub> |
| setu_no_uniform | 0.602<br><sub>[0.395, 0.854]</sub> | 1.955<br><sub>[0.208, 5.030]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 480<br><sub>[225, 705]</sub> | 0.47<br><sub>[0.31, 0.62]</sub> | 0.70<br><sub>[0.60, 0.77]</sub> | 4.2<br><sub>[3.6, 4.8]</sub> |
| **setu_full** | 0.623<br><sub>[0.408, 0.876]</sub> | 1.955<br><sub>[0.208, 5.030]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 167<br><sub>[105, 222]</sub> | 0.65<br><sub>[0.49, 0.78]</sub> | 0.85<br><sub>[0.72, 0.93]</sub> | 7.1<br><sub>[6.3, 8.0]</sub> |
| setu_no_reillum | 5.111<br><sub>[0.560, 14.037]</sub> | 8.463<br><sub>[0.269, 24.619]</sub> | 95.8%<br><sub>[87.5%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 98<br><sub>[62, 137]</sub> | 0.59<br><sub>[0.47, 0.70]</sub> | 0.84<br><sub>[0.76, 0.91]</sub> | 8.5<br><sub>[7.7, 9.5]</sub> |

**Tie-point RMSE** is the error of the delivered correspondences against exact truth, over inliers, and it is the number the problem statement's sub-pixel requirement refers to. **Model RMSE** is the error of the fitted transform; it falls roughly as the square root of the point count, so a method returning thousands of noisy points can score well on it while its individual tie points are useless. Uniformity is measured after subsampling every method to the same 150 points, because coverage of an 8x8 lattice is otherwise a measure of density rather than of distribution. Square brackets are bootstrap 95% confidence intervals over pairs.

### Ablation table

| Configuration | Tie-point RMSE (px) | Model RMSE (px) | Precision @3px | Inlier ratio | Coverage |
|---|---|---|---|---|---|
| SETU without sun-synchronised re-illumination (ablates N1) | 5.111 | 8.463 | 95.8% | 100.0% | 0.60 |
| SETU without the agreement gate, track A only (ablates N2) | 1.046 | 1.938 | 97.2% | 100.0% | 0.66 |
| SETU without sub-pixel refinement (ablates N4) | 0.961 | 1.839 | 98.6% | 91.7% | 0.76 |
| SETU without uniformity enforcement (ablates N3) | 0.602 | 1.955 | 100.0% | 100.0% | 0.55 |
| SETU, complete | 0.623 | 1.955 | 100.0% | 100.0% | 0.69 |

- **SETU without sun-synchronised re-illumination (ablates N1)**: tie-point RMSE 5.111 px against 0.623 px, a factor of 8.2.

- **SETU without the agreement gate, track A only (ablates N2)**: tie-point RMSE 1.046 px against 0.623 px, a factor of 1.7.

- **SETU without sub-pixel refinement (ablates N4)**: tie-point RMSE 0.961 px against 0.623 px, a factor of 1.5.

- **SETU without uniformity enforcement (ablates N3)**: tie-point RMSE 0.602 px against 0.623 px, a factor of 1.0.