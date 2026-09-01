# SETU

Smart India Hackathon 2026 · **Problem Statement 26166** · ISRO / Department of Space

> **Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2
> optical images (OHRC, TMC and IIRS)**

Sub-pixel correspondence between a Chandrayaan-2 image and a lunar reference map, invariant
to all three variations the statement names.

> Geometry for scale and viewpoint. Physics for the Sun. Learning only for what is left.

### The three invariances, and where each one lives

| The statement asks for | How SETU gets it | Code |
|---|---|---|
| **Multi-modal** | IIRS pseudo-panchromatic synthesis over the reflected-solar window, column destriping, a reference policy that routes each payload to a resolution-appropriate partner, and structural representations that survive non-linear radiometric change | `illum/iirs.py`, `io/registry.py`, `illum/structural.py` |
| **Sun angle invariant** | The reference is re-rendered under the source image's own solar geometry rather than searched for an invariant descriptor | `illum/render.py` |
| **Scale invariant** | Both images ortho-projected to one working GSD before matching; the coarser image is never upsampled | `geom/ortho.py`, `geom/prealign.py` |

Measured on the multi-modal axis, which is the hardest of the three: on an IIRS-class
sensor gap (low SNR, column-striped spectrometer band against a framing camera) at a 4x
scale ratio, SETU reaches **0.70 px in the source frame, 14 m on the ground**, at 64.5%
precision. The best baseline manages 311 px. SETU is the only method in the table that
registers those pairs at all.

Pushed past the design target into the thermal regime, where the source is measuring
temperature rather than reflected light, every baseline returns garbage at 0% precision
and **SETU registers none of the ten pairs**. The 900 to 1600 nm pseudo-panchromatic
window exists precisely to stay out of that regime.

---

## The problem, in one paragraph

PS 26166 asks for correspondence between a Chandrayaan-2 optical image and a lunar
reference image, with sub-pixel accuracy, a uniform distribution of match points, and
robustness to illumination, viewpoint and scale. Chandrayaan-2 products are
seleno-tagged, so the approximate footprint of every image is known before a single pixel
is matched — but that geolocation is wrong by kilometres. Independent reprocessing of
OHRC data found SPICE-projected positions roughly 4 km from the true landing site. **A
kilometre-level prior against a sub-pixel requirement is the entire problem.**

## Why the obvious approach fails, measured

On an airless body with almost no albedo variation, appearance is a property of the Sun
rather than of the surface. Two renderings of the *identical* terrain at the *identical*
viewpoint, differing only in solar azimuth:

| Comparison | Normalised cross-correlation |
|---|---|
| Same ground, sun from the east vs from the west | **−0.998** |
| The same pair, after a phase-congruency transform | **+0.80** |
| The same pair, after re-rendering the reference at the source's own Sun | **+0.97** |

Not weakly correlated — *anti*-correlated. Every gradient a descriptor keys on has changed
sign. This is why SIFT, ORB and a MegaDepth-trained network fail here rather than degrade,
and it is the fact the whole design is built around.

## The approach

SETU inverts the usual pipeline. Of the three variations the problem statement names, two
are not appearance problems at all:

- **Scale and viewpoint are geometry**, and Chandrayaan-2 ships the geometry in every
  product. Both images are ortho-projected onto one map projection at one ground sampling
  distance on SLDEM2015, which collapses scale ratios up to 160× and viewpoint differences
  of tens of degrees into a residual planar misalignment.
- **Illumination is genuine appearance**, so the reference is *re-rendered* from the
  terrain model under the exact solar azimuth, elevation and emission angle recorded in
  the Chandrayaan-2 metadata — Lunar-Lambert reflectance with McEwen limb darkening and
  ray-cast shadows. Matching then happens between two images that agree about the Sun.
- **Whatever survives** is handled by two independent matchers that only get to contribute
  a correspondence when they agree.

### Five claims, each with a number behind it

| | Claim | How it is measured |
|---|---|---|
| **N1** | Sun-synchronised reference re-illumination | Inlier ratio and RMSE against Δ(solar angle), with and without the render |
| **N2** | Agreement-gated two-track correspondence | False-match rate at fixed recall, against either track alone |
| **N3** | Uniformity as an explicit objective | Coverage, occupancy χ², Clark-Evans R — **at matched inlier counts** |
| **N4** | Per-correspondence covariance, propagated | Predicted σ against realised residual; the curve should sit on y = x |
| **N5** | Controlled benchmark with exact ground truth | Registration error measured against a known warp, not a fitted one |

## Architecture

```
S0  Ingest              PDS4 / PDS3 / GeoTIFF behind one Product type; illumination
                        resolved from backplane → SPICE → label keywords → fail loudly
S1  Pre-alignment       common CRS, common GSD, ortho-projection on SLDEM2015
S2  Illumination        (a) re-render the reference at the source's Sun        ← N1
                        (b) phase congruency / MIM / CFOG / LNIFT
                        (c) IIRS pseudo-panchromatic synthesis
S3  Correspondence      A: dense deep matcher, tiled
                        B: PC detection + MIM description + CFOG template
                        → agreement gate                                        ← N2
S4  Refinement          upsampled phase correlation + LSM + covariance          ← N4
S5  Model               MAGSAC++ → local TPS/polynomial → per-row jitter spline
S6  Uniformity          lattice quota + farthest-point selection + re-seeding   ← N3
S7  Products            GeoTIFF | tiepoints CSV+GeoJSON | PDS4 label | QA report
S8  Evaluation          metrics | baselines | sweeps | leaderboard              ← N5

     S5 → S1   re-project from the corrected footprint (one iteration, two is the cap)
     S6 → S3   re-seed failed cells at a lowered threshold and a ±5 px window
```

Two things about this order are deliberate. **S8 was built before S3** — the harness is
the credibility, and building the matcher first would have meant tuning it against
intuition. And **both feedback edges are implemented**, because the re-seeding edge is
what actually delivers the uniformity the problem statement asks for.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"
pip install torch torchvision kornia     # optional: enables the deep track

setu info                                # what this environment can actually do
pytest -q                                # 40 tests, including a full end-to-end run
```

Register a real pair:

```bash
setu register \
  --source data/ch2/ohrc/ch2_ohr_ncp_2024...xml \
  --reference data/lro/nac/M1234567890LE.IMG \
  --dem data/sldem/sldem2015_tile.tif \
  --config configs/ohrc_nac.yaml \
  --out runs/2026-09-05_ohrc_nac_01
```

Reproduce every number on the demo site:

```bash
python experiments/run_sweeps.py        # the three sweeps and the leaderboard
python scripts/build_demo_bundle.py     # the web demo's data and imagery
```

Or in Docker, which is the reproducible path:

```bash
docker build -t setu . && docker run -p 8000:8000 setu
```

## What a run produces

| File | Contents |
|---|---|
| `registered.tif` | Source resampled into the reference geometry, cloud-optimised, Lanczos |
| `tiepoints.csv` | Per point: pixel and selenographic coordinates, confidence, track, **σx σy σxy**, residuals, inlier and re-seed flags, lattice cell |
| `tiepoints.geojson` | The same list, ready for QGIS |
| `transform.json` | Global matrix, local model, per-row jitter spline, CRS, GSD, resolved config |
| `metrics.json` | Every metric in the protocol, plus per-stage timing |
| `label.xml` | PDS4-style label carrying both product IDs and the illumination state |
| `report.html` | Self-contained QA report, every image embedded |

## On honesty

Three rules the code enforces on itself, because registration results are unusually easy
to overstate:

**A fit residual is not an accuracy.** `metrics.json` reports the model's residual on its
own inliers and labels it as exactly that. The number that answers the problem statement's
sub-pixel requirement is the *per-tie-point* error, which is larger. Both appear in every
table, under different names.

**A local model must earn its place.** A thin-plate spline with too little regularisation
interpolates its control points exactly and reports a fit RMSE of 0.000000 while
generalising at 0.117 px. The regularisation is chosen by leave-one-out cross-validation
and the LOOCV RMSE is a mandatory output, not an option.

**Uniformity is compared at matched point counts.** A semi-dense matcher returning 3,000
points covers every cell of an 8×8 lattice whatever its spatial behaviour. Comparing raw
coverage would measure density and call it distribution, so every method is subsampled to
the same count before its uniformity is scored.

## Data

The problem statement's dataset is marked TBD. `scripts/manifest.yaml` records the archive
that does exist — Chandrayaan-2 L1 via PRADAN, LRO NAC and NAC DTMs, SLDEM2015, Kaguya TC,
LROC WAC — and `scripts/fetch_data.py` fetches what can be fetched and prints precise
instructions for the archives that need an account.

Results published here are measured on the **controlled benchmark**, where both images of
every pair are rendered from one terrain model under a transform that is known exactly, so
registration error is true geometric error rather than the residual of a fitted model. The
PDS4, PDS3 and GeoTIFF readers, the SPICE path, the sensor models and the reference policy
are all implemented and exercised; they are waiting on archive access, not on code. This
is the sequencing the specification's own risk table recommends: build against the
synthetic bench so that no phase is blocked on a download.

## Layout

```
setu/
├── configs/            default + one YAML per experiment
├── setu/
│   ├── types.py        Product, IlluminationState, TiePoint, RunResult
│   ├── io/             pds4, pds3, geotiff, isis, registry + REFERENCE_POLICY
│   ├── geom/           crs, sensor_model (Tier A/B), prealign, ortho
│   ├── illum/          render (N1), reflectance, shadow, structural, iirs
│   ├── match/          base, deep, structural, tiling, gate (N2)
│   ├── refine/         phasecorr, lsm, covariance (N4)
│   ├── model/          robust, local, jitter
│   ├── uniform/        lattice, anms, reseed, stats (N3)
│   ├── product/        warp, writers, pds4_label, report
│   ├── bench/          terrain, generate (N5), degrade
│   ├── eval/           metrics, runner, baselines, plots, leaderboard
│   ├── pipeline.py     S0–S7 with both feedback edges
│   └── cli.py          register | bench | eval | serve | info
├── api/                FastAPI service
├── web/                React + Vite demo
├── experiments/        the sweeps that produce the published numbers
└── tests/              metrics, reflectance, uniformity, end-to-end
```

## References

1. He et al. *MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training.* TPAMI 2026, arXiv:2501.07556
2. Li, Hu & Ai. *RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform.* IEEE TIP 29:3296, 2020
3. Ye et al. *Robust registration of multimodal remote sensing images* (HOPC), IEEE TGRS 55(5), 2017; and *CFOG*, ISPRS 2019
4. Kovesi. *Image Features from Phase Congruency.* 1999
5. Kumar, Kaushal & Murthy. *MoonMetaSync: Lunar Image Registration Analysis.* arXiv:2410.11118, 2024
6. Tungathurthi. *Geodetically Anchored 0.30 m DEM of the Chandrayaan-3 Vikram Landing Site.* arXiv:2602.14993, 2026
7. Barker et al. *A new lunar digital elevation model from LOLA and SELENE Terrain Camera* (SLDEM2015). Icarus 273:346, 2016
8. Henriksen et al. *Extracting accurate and precise topography from LROC NAC stereo observations.* Icarus 283:122, 2017
9. Chowdhury et al. *Chandrayaan-2 Orbiter High Resolution Camera.* Current Science 118(4):560, 2020
10. Verma, Chauhan & Chauhan. *Lunar surface temperature estimation and thermal emission correction using Chandrayaan-2 IIRS data.* Icarus 383:115075, 2022
11. *Robust feature matching of multi-illumination lunar orbiter images based on crater neighbourhood structure.* Remote Sensing 17(13):2302, 2025
12. Guizar-Sicairos et al. *Efficient subpixel image registration algorithms* (upsampled DFT cross-correlation)
13. Barath et al. *MAGSAC++.* CVPR 2020
14. Beyer, Alexandrov & McMichael. *The Ames Stereo Pipeline.* Earth and Space Science 5:537, 2018
