### Azimuth sweep (delta sun azimuth 0-180 deg, elevation fixed)

| Method | Tie-point RMSE (px) | Model RMSE vs truth (px) | Precision @3px | Inlier ratio | Inliers | Coverage @150 pts | Clark-Evans R @150 | Time (s) |
|---|---|---|---|---|---|---|---|---|
| setu_no_uniform | 0.351<br><sub>[0.310, 0.392]</sub> | 0.048<br><sub>[0.038, 0.057]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 881<br><sub>[747, 1024]</sub> | 0.68<br><sub>[0.63, 0.73]</sub> | 0.75<br><sub>[0.71, 0.78]</sub> | 4.5<br><sub>[4.1, 4.9]</sub> |
| **setu_full** | 0.361<br><sub>[0.320, 0.407]</sub> | 0.048<br><sub>[0.038, 0.057]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 245<br><sub>[227, 265]</sub> | 0.81<br><sub>[0.78, 0.85]</sub> | 0.90<br><sub>[0.87, 0.94]</sub> | 5.6<br><sub>[5.3, 5.9]</sub> |
| setu_no_gate | 0.362<br><sub>[0.312, 0.413]</sub> | 0.059<br><sub>[0.046, 0.076]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 208<br><sub>[192, 224]</sub> | 0.78<br><sub>[0.75, 0.81]</sub> | 0.81<br><sub>[0.78, 0.84]</sub> | 4.5<br><sub>[4.2, 4.8]</sub> |
| setu_no_refine | 0.304<br><sub>[0.270, 0.337]</sub> | 0.068<br><sub>[0.056, 0.081]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 100.0%<br><sub>[100.0%, 100.0%]</sub> | 258<br><sub>[241, 277]</sub> | 0.83<br><sub>[0.80, 0.86]</sub> | 0.91<br><sub>[0.88, 0.93]</sub> | 3.7<br><sub>[3.5, 3.8]</sub> |
| setu_no_reillum <sub>(5/14 failed)</sub> | 0.614<br><sub>[0.383, 0.874]</sub> | 0.345<br><sub>[0.069, 0.819]</sub> | 88.9%<br><sub>[66.7%, 100.0%]</sub> | 55.6%<br><sub>[22.2%, 88.9%]</sub> | 63<br><sub>[15, 124]</sub> | 0.66<br><sub>[0.42, 0.85]</sub> | 0.93<br><sub>[0.87, 0.99]</sub> | 4.3<br><sub>[2.5, 6.1]</sub> |
| cfog | 1.751<br><sub>[1.550, 2.014]</sub> | 1.449<br><sub>[1.071, 1.823]</sub> | 53.2%<br><sub>[38.4%, 66.0%]</sub> | 55.3%<br><sub>[40.4%, 68.4%]</sub> | 142<br><sub>[103, 175]</sub> | 0.65<br><sub>[0.53, 0.75]</sub> | 1.16<br><sub>[1.06, 1.24]</sub> | 0.4<br><sub>[0.4, 0.4]</sub> |
| loftr | 10.247<br><sub>[3.654, 18.544]</sub> | 16.056<br><sub>[5.284, 29.135]</sub> | 35.9%<br><sub>[16.2%, 56.9%]</sub> | 38.0%<br><sub>[19.0%, 58.2%]</sub> | 702<br><sub>[209, 1349]</sub> | 0.40<br><sub>[0.21, 0.59]</sub> | 0.60<br><sub>[0.44, 0.74]</sub> | 0.4<br><sub>[0.4, 0.5]</sub> |
| orb | 83.116<br><sub>[41.174, 124.059]</sub> | 212.806<br><sub>[133.716, 293.526]</sub> | 18.1%<br><sub>[3.4%, 37.9%]</sub> | 31.3%<br><sub>[18.6%, 46.9%]</sub> | 178<br><sub>[8, 480]</sub> | 0.13<br><sub>[0.06, 0.23]</sub> | 0.34<br><sub>[0.22, 0.45]</sub> | 0.0<br><sub>[0.0, 0.0]</sub> |
| rift | 91.531<br><sub>[39.090, 144.628]</sub> | 229.138<br><sub>[68.597, 453.573]</sub> | 34.6%<br><sub>[14.8%, 56.7%]</sub> | 51.9%<br><sub>[36.9%, 67.9%]</sub> | 62<br><sub>[14, 121]</sub> | 0.22<br><sub>[0.10, 0.34]</sub> | 0.83<br><sub>[0.70, 0.98]</sub> | 0.6<br><sub>[0.6, 0.7]</sub> |
| intfeat | 145.432<br><sub>[91.528, 205.626]</sub> | 234.834<br><sub>[154.936, 310.375]</sub> | 14.0%<br><sub>[1.2%, 31.8%]</sub> | 26.7%<br><sub>[14.9%, 41.4%]</sub> | 191<br><sub>[7, 520]</sub> | 0.14<br><sub>[0.06, 0.26]</sub> | 0.47<br><sub>[0.30, 0.69]</sub> | 0.1<br><sub>[0.1, 0.2]</sub> |
| disk_lightglue | 113.813<br><sub>[47.872, 181.830]</sub> | 356.101<br><sub>[86.955, 741.446]</sub> | 37.4%<br><sub>[15.7%, 60.1%]</sub> | 56.4%<br><sub>[40.9%, 71.7%]</sub> | 513<br><sub>[191, 866]</sub> | 0.31<br><sub>[0.14, 0.47]</sub> | 0.69<br><sub>[0.54, 0.82]</sub> | 0.5<br><sub>[0.4, 0.6]</sub> |
| sift | 150.123<br><sub>[91.452, 202.186]</sub> | 407.060<br><sub>[233.832, 615.374]</sub> | 18.9%<br><sub>[2.9%, 39.0%]</sub> | 41.7%<br><sub>[30.2%, 55.1%]</sub> | 80<br><sub>[4, 223]</sub> | 0.14<br><sub>[0.05, 0.26]</sub> | 0.87<br><sub>[0.58, 1.18]</sub> | 0.0<br><sub>[0.0, 0.0]</sub> |

**Tie-point RMSE** is the error of the delivered correspondences against exact truth, over inliers, and it is the number the problem statement's sub-pixel requirement refers to. **Model RMSE** is the error of the fitted transform; it falls roughly as the square root of the point count, so a method returning thousands of noisy points can score well on it while its individual tie points are useless. Uniformity is measured after subsampling every method to the same 150 points, because coverage of an 8x8 lattice is otherwise a measure of density rather than of distribution. Square brackets are bootstrap 95% confidence intervals over pairs.

### Ablation table

| Configuration | Tie-point RMSE (px) | Model RMSE (px) | Precision @3px | Inlier ratio | Coverage |
|---|---|---|---|---|---|
| SETU without sun-synchronised re-illumination (ablates N1) | 0.614 | 0.345 | 88.9% | 55.6% | 0.68 |
| SETU without the agreement gate, track A only (ablates N2) | 0.362 | 0.059 | 100.0% | 100.0% | 0.84 |
| SETU without sub-pixel refinement (ablates N4) | 0.304 | 0.068 | 100.0% | 100.0% | 0.89 |
| SETU without uniformity enforcement (ablates N3) | 0.351 | 0.048 | 100.0% | 100.0% | 0.82 |
| SETU, complete | 0.361 | 0.048 | 100.0% | 100.0% | 0.86 |

- **SETU without sun-synchronised re-illumination (ablates N1)**: tie-point RMSE 0.614 px against 0.361 px, a factor of 1.7.

- **SETU without the agreement gate, track A only (ablates N2)**: tie-point RMSE 0.362 px against 0.361 px, a factor of 1.0.

- **SETU without sub-pixel refinement (ablates N4)**: tie-point RMSE 0.304 px against 0.361 px, a factor of 0.8.

- **SETU without uniformity enforcement (ablates N3)**: tie-point RMSE 0.351 px against 0.361 px, a factor of 1.0.