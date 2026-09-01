import { useRef, useState } from 'react'
import { useReveal } from '../lib/reveal'
import { fmt, pct, smart, type EvalBundle, type MethodSummary } from '../lib/data'

const base = import.meta.env.BASE_URL || '/'
const url = (p: string) => `${base}${p}`.replace(/\/{2,}/g, '/')

const ORDER = [
  'sift', 'orb', 'intfeat', 'rift', 'cfog',
  'disk_lightglue', 'loftr', 'matchanything_eloftr',
  'setu_no_reillum', 'setu_no_gate', 'setu_no_refine', 'setu_no_uniform', 'setu_full',
]

const NAMES: Record<string, string> = {
  sift: 'SIFT + FLANN',
  orb: 'ORB + BF',
  intfeat: 'IntFeat (MoonMetaSync)',
  rift: 'RIFT (PC + MIM)',
  cfog: 'CFOG template matching',
  disk_lightglue: 'DISK + LightGlue',
  loftr: 'LoFTR',
  matchanything_eloftr: 'MatchAnything ELoFTR',
  setu_no_reillum: 'SETU − re-illumination (N1 off)',
  setu_no_gate: 'SETU − agreement gate (N2 off)',
  setu_no_refine: 'SETU − sub-pixel refinement (N4 off)',
  setu_no_uniform: 'SETU − uniformity (N3 off)',
  setu_full: 'SETU, complete',
}

const SWEEPS = [
  { key: 'summary_azimuth', label: 'Solar azimuth 0 to 180°', plot: 'rmse_vs_sun_azimuth.png',
    caption: 'Solar elevation held at 25°, so shadow length is identical on both sides and only the Sun’s direction changes. This is the cleanest isolation of the illumination problem.' },
  { key: 'summary_illumination', label: 'Solar elevation 10 to 75°', plot: 'rmse_vs_sun_elevation.png',
    caption: 'Source elevation swept against a fixed 45° reference. Absolute elevation is an unavoidable confound here, and it controls how much shadow exists to match on at all.' },
  { key: 'summary_scale', label: 'Scale ratio 1 to 16×', plot: 'rmse_vs_scale.png',
    caption: 'Ground sampling ratio swept at fixed illumination. Pre-alignment absorbs the ratio before any matcher sees the pair.' },
  { key: 'summary_multimodal', label: 'Multi-modal', plot: 'rmse_reflected_solar.png',
    caption: 'An IIRS-class sensor gap at a 4× scale ratio: a low-SNR, column-striped spectrometer band against a framing camera. SETU is the only method here that registers the pairs at all.' },
] as const

function Cell({ s, digits = 3, kind = 'num' }: { s?: { value: number; ci_lo: number; ci_hi: number; n: number }; digits?: number; kind?: string }) {
  if (!s || !Number.isFinite(s.value)) return <td className="n">—</td>
  const f = kind === 'pct' ? pct : kind === 'smart' ? smart : (v: number) => fmt(v, digits)
  return (
    <td className="n">
      {f(s.value)}
      {s.n > 1 && Number.isFinite(s.ci_lo) && (
        <span className="ci">[{f(s.ci_lo)}, {f(s.ci_hi)}]</span>
      )}
    </td>
  )
}

export default function Results({ evalData }: { evalData: EvalBundle | null }) {
  const scope = useRef<HTMLElement>(null)
  const [sweep, setSweep] = useState(0)

  useReveal('.r-reveal', [evalData, sweep])

  const current = SWEEPS[sweep]
  const summary: Record<string, MethodSummary> =
    (evalData?.[current.key as keyof EvalBundle] as any) ?? evalData?.summary_all ?? {}
  const methods = ORDER.filter((m) => m in summary)
  const full = summary.setu_full

  return (
    <section id="results" ref={scope}>
      <div className="wrap">
        <p className="eyebrow r-reveal">Measured, not asserted</p>
        <h2 className="r-reveal">Every number here came out of the evaluation harness</h2>
        <p className="lede r-reveal">
          Both images of every benchmark pair are rendered from one terrain model under a
          transform that is known exactly, so registration error is true geometric error
          rather than the residual of a model fitted to the data. Each method runs through
          the same robust fit and the same metric code; only the correspondence step
          differs.
        </p>

        {!evalData && (
          <div className="note r-reveal" style={{ marginTop: 28 }}>
            Evaluation data has not been built yet. Run{' '}
            <code className="mono">python experiments/run_sweeps.py</code> then{' '}
            <code className="mono">python scripts/build_demo_bundle.py</code>.
          </div>
        )}

        {evalData && (
          <>
            {full && (
              <div className="grid g4 r-reveal" style={{ marginTop: 40 }}>
                <div className="stat panel accent">
                  <div className="k">Tie-point RMSE</div>
                  <div className="v">{fmt(full.rmse_inliers_px?.value, 3)}</div>
                  <div className="u">px against exact truth</div>
                </div>
                <div className="stat panel good">
                  <div className="k">Precision @3px</div>
                  <div className="v">{pct(full.precision_3px?.value, 1)}</div>
                  <div className="u">correspondences within 3 px of truth</div>
                </div>
                <div className="stat panel">
                  <div className="k">Inlier ratio</div>
                  <div className="v">{pct(full.inlier_ratio?.value, 1)}</div>
                  <div className="u">after MAGSAC++</div>
                </div>
                <div className="stat panel">
                  <div className="k">Coverage @150 pts</div>
                  <div className="v">{fmt(full.coverage_matched?.value, 2)}</div>
                  <div className="u">R = {fmt(full.clark_evans_matched?.value, 2)} at matched count</div>
                </div>
              </div>
            )}

            <div className="sweep-tabs r-reveal">
              {SWEEPS.map((s, i) => (
                <button key={s.key} className={`sweep-tab${i === sweep ? ' on' : ''}`} onClick={() => setSweep(i)}>
                  {s.label}
                </button>
              ))}
            </div>

            <div className="grid g2 r-reveal" style={{ alignItems: 'start' }}>
              <figure className="panel fig wide">
                <img src={url(`demo/${current.plot}`)} alt={current.label} />
                <figcaption>{current.caption}</figcaption>
              </figure>
              <div className="note" style={{ alignSelf: 'center' }}>
                <strong>Reading the two columns.</strong> Tie-point RMSE is the error of the
                delivered correspondences and is what the problem statement&rsquo;s sub-pixel
                requirement refers to. Model RMSE is the error of the fitted transform, and it
                falls roughly as the square root of the point count &mdash; so a method that
                returns three thousand noisy points can score well on it while its individual
                tie points are unusable. Both are shown, because only reporting the flattering
                one is how registration results get overstated.
              </div>
            </div>

            <div className="panel table-scroll r-reveal" style={{ marginTop: 22 }}>
              <table>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th style={{ textAlign: 'right' }}>Tie-point RMSE (px)</th>
                    <th style={{ textAlign: 'right' }}>Model RMSE (px)</th>
                    <th style={{ textAlign: 'right' }}>Precision @3px</th>
                    <th style={{ textAlign: 'right' }}>Inlier ratio</th>
                    <th style={{ textAlign: 'right' }}>Inliers</th>
                    <th style={{ textAlign: 'right' }}>Coverage @150</th>
                    <th style={{ textAlign: 'right' }}>R @150</th>
                    <th style={{ textAlign: 'right' }}>Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {methods.map((m) => {
                    const e = summary[m]
                    const failed = e.n_ok === 0
                    return (
                      <tr key={m} className={m === 'setu_full' ? 'hi' : ''}>
                        <td>
                          {NAMES[m] ?? m}
                          {e.n_failed > 0 && (
                            <span className="tag warn" style={{ marginLeft: 8 }}>
                              {e.n_failed}/{e.n_pairs} failed
                            </span>
                          )}
                        </td>
                        {failed ? (
                          <td className="n" colSpan={8} style={{ color: 'var(--bad)' }}>no registration</td>
                        ) : (
                          <>
                            <Cell s={e.rmse_inliers_px} kind="smart" />
                            <Cell s={e.rmse_true_px} kind="smart" />
                            <Cell s={e.precision_3px} kind="pct" />
                            <Cell s={e.inlier_ratio} kind="pct" />
                            <Cell s={e.n_inliers} digits={0} />
                            <Cell s={e.coverage_matched} digits={2} />
                            <Cell s={e.clark_evans_matched} digits={2} />
                            <Cell s={e.seconds} digits={1} />
                          </>
                        )}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <p className="tbl-note r-reveal">
              Square brackets are bootstrap 95% confidence intervals over pairs.
              Uniformity is measured after subsampling every method to the same 150 points,
              because coverage of an 8&times;8 lattice otherwise measures density rather than
              distribution. Benchmark generated from {evalData.n_pairs} pairs.
            </p>
          </>
        )}
      </div>
    </section>
  )
}
