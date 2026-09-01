/**
 * Data the page renders.
 *
 * Every number on this site comes from a file written by `setu/eval/` — the leaderboard
 * JSON, a run's `metrics.json`, or the demo bundle produced by
 * `scripts/build_demo_bundle.py`. Nothing is typed in by hand, because a figure on a slide
 * that cannot be traced back to a run is exactly what the specification forbids.
 */

export interface Stat {
  value: number
  ci_lo: number
  ci_hi: number
  n: number
}

export interface MethodSummary {
  label: string
  n_pairs: number
  n_ok: number
  n_failed: number
  rmse_inliers_px: Stat
  rmse_true_px: Stat
  rmse_points_px: Stat
  precision_3px: Stat
  inlier_ratio: Stat
  n_inliers: Stat
  coverage: Stat
  coverage_matched: Stat
  clark_evans_matched: Stat
  clark_evans_R: Stat
  seconds: Stat
  [key: string]: unknown
}

export interface SweepPoint {
  method: string
  pair_id: string
  d_sun_elev: number
  d_sun_az: number
  scale_ratio: number
  rmse_inliers_px: number
  rmse_true_px: number
  inlier_ratio: number
  coverage: number
  n_inliers: number
  ok: boolean
}

export interface EvalBundle {
  generated_utc: string
  n_pairs: number
  methods: string[]
  summary_all: Record<string, MethodSummary>
  summary_azimuth?: Record<string, MethodSummary>
  summary_illumination?: Record<string, MethodSummary>
  summary_scale?: Record<string, MethodSummary>
  summary_multimodal?: Record<string, MethodSummary>
  summary_multimodal_thermal?: Record<string, MethodSummary>
  rows?: SweepPoint[]
}

export interface StageRecord {
  stage: string
  label: string
  seconds: number
  [key: string]: unknown
}

export interface DemoRun {
  key: string
  label: string
  blurb: string
  run_id: string
  source: { pid: string; sensor: string; gsd_m: number; illumination: Record<string, number | string> }
  reference: { pid: string; sensor: string; gsd_m: number; illumination: Record<string, number | string> }
  metrics: Record<string, any>
  stages: StageRecord[]
  images: Record<string, string>
  tiepoints: Array<{
    x: number; y: number; rx: number; ry: number
    r: number; sx: number; conf: number; track: string; inlier: boolean; reseeded: boolean
  }>
  truth: {
    ncc_real: number | null
    ncc_rendered: number | null
    d_sun_elev: number
    d_sun_az: number
    scale_ratio: number
    gsd_src_m: number
    gsd_ref_m: number
  }
}

export interface IllumDemo {
  images: Record<string, string>
  ncc_opposite_azimuth: number
  ncc_elevation_change: number
  ncc_structural_opposite: number
  shadow_fraction_low_sun: number
  shadow_fraction_high_sun: number
}

export interface DemoBundle {
  generated_utc: string
  illumination: IllumDemo
  scenes: DemoRun[]
}

const base = import.meta.env.BASE_URL || '/'

async function loadJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${base}${path}`.replace(/\/{2,}/g, '/'))
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

export const loadEval = () => loadJson<EvalBundle>('demo/eval.json')
export const loadDemo = () => loadJson<DemoBundle>('demo/demo.json')

export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(digits)

export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(digits)}%`

/** Compact rendering for numbers that span several orders of magnitude. */
export const smart = (v: number | null | undefined): string => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (v >= 1000) return v.toExponential(1)
  if (v >= 100) return v.toFixed(0)
  if (v >= 10) return v.toFixed(1)
  return v.toFixed(3)
}
