import { useEffect, useMemo, useRef, useState } from 'react'
import { prefersReducedMotion } from '../lib/motion'
import { fmt, pct, type DemoRun } from '../lib/data'
import { useReveal } from '../lib/reveal'

const base = import.meta.env.BASE_URL || '/'
const url = (p?: string) => (p ? `${base}${p}`.replace(/\/{2,}/g, '/') : undefined)

/** Compass rosette showing where the Sun was for one acquisition. */
function SunRosette({ az, elev, label, color }: { az: number; elev: number; label: string; color: string }) {
  const r = 30
  const a = ((az - 90) * Math.PI) / 180
  const len = r * (1 - Math.min(elev, 90) / 110)
  return (
    <div className="rosette">
      <svg viewBox="-40 -40 80 80" width="78" height="78">
        <circle r={r} fill="none" stroke="var(--line-2)" strokeWidth="1" />
        <circle r={r * 0.55} fill="none" stroke="var(--line)" strokeWidth="1" strokeDasharray="2 3" />
        <line x1="0" y1={-r} x2="0" y2={r} stroke="var(--line)" strokeWidth="0.6" />
        <line x1={-r} y1="0" x2={r} y2="0" stroke="var(--line)" strokeWidth="0.6" />
        <text x="0" y={-r - 4} fontSize="7" fill="var(--dim)" textAnchor="middle">N</text>
        <line x1="0" y1="0" x2={Math.cos(a) * len} y2={Math.sin(a) * len} stroke={color} strokeWidth="2" />
        <circle cx={Math.cos(a) * len} cy={Math.sin(a) * len} r="3.6" fill={color} />
        <circle r="2" fill="var(--muted)" />
      </svg>
      <div className="rosette-cap">
        <strong>{label}</strong>
        <span>az {az.toFixed(0)}&deg; · el {elev.toFixed(0)}&deg;</span>
      </div>
    </div>
  )
}

/** Tie points plotted over the source frame, coloured by residual. */
function TiePointMap({ scene }: { scene: DemoRun }) {
  const pts = scene.tiepoints.filter((t) => t.inlier)
  const max = Math.max(0.5, ...pts.map((t) => t.r))
  const colour = (r: number) => {
    const t = Math.min(r / max, 1)
    // cyan (accurate) through amber to red (worst residual in this run)
    return t < 0.5
      ? `rgb(${34 + t * 2 * 200},${211 - t * 2 * 53},${238 - t * 2 * 227})`
      : `rgb(${234},${158 - (t - 0.5) * 2 * 45},${11 + (t - 0.5) * 2 * 60})`
  }
  return (
    <div className="tp-map">
      <img src={url(scene.images.source)} alt="Source image" />
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
          <g key={i} stroke="rgba(34,211,238,.22)" strokeWidth="0.15">
            <line x1={(i * 100) / 8} y1="0" x2={(i * 100) / 8} y2="100" />
            <line x1="0" y1={(i * 100) / 8} x2="100" y2={(i * 100) / 8} />
          </g>
        ))}
        {pts.map((t, i) => (
          <circle
            key={i}
            cx={t.x * 100}
            cy={t.y * 100}
            r={t.reseeded ? 1.0 : 0.85}
            fill={colour(t.r)}
            stroke={t.reseeded ? '#a78bfa' : 'rgba(0,0,0,.6)'}
            strokeWidth={t.reseeded ? 0.35 : 0.18}
          />
        ))}
      </svg>
      <div className="tp-legend">
        <span><i style={{ background: '#22d3ee' }} /> 0 px</span>
        <span><i style={{ background: '#ea9e0b' }} /> {fmt(max / 2, 2)} px</span>
        <span><i style={{ background: '#ea6a47' }} /> {fmt(max, 2)} px</span>
        <span><i style={{ background: 'transparent', border: '1.5px solid #a78bfa' }} /> re-seeded</span>
      </div>
    </div>
  )
}

/** Before / after swipe over the reference. */
function Swipe({ before, after }: { before?: string; after?: string }) {
  const [pos, setPos] = useState(50)
  return (
    <div className="swipe">
      <img className="swipe-b" src={url(after)} alt="After registration" />
      <div className="swipe-clip" style={{ width: `${pos}%` }}>
        <img src={url(before)} alt="Before registration" />
      </div>
      <div className="swipe-handle" style={{ left: `${pos}%` }}>
        <i />
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        aria-label="Compare before and after registration"
        onChange={(e) => setPos(Number(e.target.value))}
      />
      <span className="swipe-tag left">before</span>
      <span className="swipe-tag right">after</span>
    </div>
  )
}

export default function Demo({ scenes }: { scenes: DemoRun[] }) {
  const scope = useRef<HTMLElement>(null)
  const [active, setActive] = useState(0)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const scene = scenes[active]
  const stages = useMemo(
    () => (scene?.stages ?? []).filter((s) => s.stage !== 'S0'),
    [scene],
  )

  // Staged playback: the re-illumination reveal is a visible step, not a log line.
  useEffect(() => {
    if (!playing) return
    if (step >= stages.length) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setStep((s) => s + 1), prefersReducedMotion() ? 120 : 620)
    return () => clearTimeout(t)
  }, [playing, step, stages.length])

  useReveal('.d-reveal', [scenes, active])

  if (!scene) {
    return (
      <section id="demo" ref={scope}>
        <div className="wrap">
          <p className="eyebrow">Demo</p>
          <h2>Registration, end to end</h2>
          <p className="lede">
            Demo data has not been built yet. Run{' '}
            <code className="mono">python scripts/build_demo_bundle.py</code> to produce it.
          </p>
        </div>
      </section>
    )
  }

  const m = scene.metrics
  const u = m.uniformity ?? {}
  const si = scene.source.illumination as Record<string, number>
  const ri = scene.reference.illumination as Record<string, number>
  const done = (i: number) => !playing || step > i

  return (
    <section id="demo" ref={scope}>
      <div className="wrap">
        <p className="eyebrow d-reveal">The demo</p>
        <h2 className="d-reveal">Three panels, one page</h2>
        <p className="lede d-reveal">
          Every scene below is a complete run of the pipeline against a pair whose true
          geometric relationship is known exactly, so the accuracy shown is measured
          against truth rather than against the model that was just fitted.
        </p>

        <div className="scene-tabs d-reveal">
          {scenes.map((s, i) => (
            <button
              key={s.key ?? i}
              className={`scene-tab${i === active ? ' on' : ''}`}
              onClick={() => { setActive(i); setStep(0); setPlaying(false) }}
            >
              <span className="scene-tab-label">{s.label}</span>
              <span className="scene-tab-sub">
                &Delta;az {s.truth.d_sun_az}&deg; · &Delta;el {s.truth.d_sun_elev}&deg; ·{' '}
                {s.truth.scale_ratio}&times; scale
              </span>
            </button>
          ))}
        </div>

        <p className="scene-blurb d-reveal">{scene.blurb}</p>

        {/* ---------------------------------------------------------- panel 1 */}
        <div className="dpanel d-reveal">
          <div className="dpanel-head">
            <span className="dpanel-n">01</span>
            <h3>Input</h3>
            <p>Two products, and the Sun they were each taken under.</p>
          </div>
          <div className="grid g2">
            <figure className="panel fig">
              <img src={url(scene.images.source)} alt="Source product" />
              <figcaption>
                Source · {scene.source.sensor} · {scene.truth.gsd_src_m} m/px
              </figcaption>
            </figure>
            <figure className="panel fig">
              <img src={url(scene.images.reference)} alt="Reference product" />
              <figcaption>
                Reference · {scene.reference.sensor} · {scene.truth.gsd_ref_m} m/px
              </figcaption>
            </figure>
          </div>
          <div className="rosette-row">
            <SunRosette az={Number(si.sun_az_deg)} elev={Number(si.sun_elev_deg)} label="Source" color="var(--sun)" />
            <SunRosette az={Number(ri.sun_az_deg)} elev={Number(ri.sun_elev_deg)} label="Reference" color="var(--accent)" />
            <div className="rosette-note">
              The two acquisitions are <strong>{scene.truth.d_sun_az}&deg;</strong> apart in
              solar azimuth and <strong>{scene.truth.d_sun_elev}&deg;</strong> apart in
              elevation. One glance at the rosettes is the whole problem.
            </div>
          </div>
        </div>

        {/* ---------------------------------------------------------- panel 2 */}
        <div className="dpanel d-reveal">
          <div className="dpanel-head">
            <span className="dpanel-n">02</span>
            <h3>Process</h3>
            <p>Staged, with the re-illumination as a visible step.</p>
            <button className="pill solid" onClick={() => { setStep(0); setPlaying(true) }}>
              {playing ? 'Running…' : 'Play the pipeline'}
            </button>
          </div>

          <ol className="stage-list">
            {stages.map((s, i) => (
              <li key={s.stage} className={done(i) ? 'on' : ''}>
                <span className="s-id">{s.stage}</span>
                <span className="s-label">{s.label}</span>
                <span className="s-time mono">{done(i) ? `${s.seconds.toFixed(2)}s` : '·'}</span>
              </li>
            ))}
          </ol>

          <div className="reveal-strip">
            <figure className="panel fig">
              <img src={url(scene.images.reference)} alt="Real reference" />
              <figcaption>The real reference, at its own Sun</figcaption>
            </figure>
            <div className="arrow">
              <span>re-rendered at the source&rsquo;s Sun</span>
              <svg viewBox="0 0 60 12" width="60" height="12">
                <path d="M0 6 H52 M46 2 L52 6 L46 10" stroke="var(--accent)" strokeWidth="1.2" fill="none" />
              </svg>
            </div>
            <figure className="panel fig accent-border">
              <img src={url(scene.images.rendered)} alt="Reference re-rendered at the source's solar geometry" />
              <figcaption>The synthetic reference · novelty N1</figcaption>
            </figure>
          </div>

          {scene.truth.ncc_real !== undefined && scene.truth.ncc_real !== null && (
            <div className="grid g2" style={{ marginTop: 16 }}>
              <div className="stat panel">
                <div className="k">Correlation with the real reference</div>
                <div className="v">{fmt(scene.truth.ncc_real, 3)}</div>
                <div className="u">before re-illumination</div>
              </div>
              <div className="stat panel accent">
                <div className="k">Correlation with the rendered reference</div>
                <div className="v">{fmt(scene.truth.ncc_rendered, 3)}</div>
                <div className="u">after re-illumination · same ground, same Sun</div>
              </div>
            </div>
          )}
        </div>

        {/* ---------------------------------------------------------- panel 3 */}
        <div className="dpanel d-reveal">
          <div className="dpanel-head">
            <span className="dpanel-n">03</span>
            <h3>Result</h3>
            <p>Measured against exact ground truth.</p>
          </div>

          <div className="grid g4">
            <div className="stat panel accent">
              <div className="k">Tie-point RMSE vs truth</div>
              <div className="v">{fmt(m.rmse_vs_truth_px, 3)}</div>
              <div className="u">px · {fmt(m.rmse_vs_truth_m, 2)} m on the ground</div>
            </div>
            <div className="stat panel">
              <div className="k">Inliers</div>
              <div className="v">{m.n_inliers}</div>
              <div className="u">{pct(m.inlier_ratio)} of putative matches</div>
            </div>
            <div className="stat panel">
              <div className="k">Median &sigma; per point</div>
              <div className="v">{fmt(m.median_sigma_px, 3)}</div>
              <div className="u">px · propagated into the fit</div>
            </div>
            <div className="stat panel">
              <div className="k">Cell coverage</div>
              <div className="v">{fmt(u.coverage_ratio, 2)}</div>
              <div className="u">
                of {u.n_valid_cells ?? '—'} cells · R = {fmt(u.clark_evans_R, 2)}
              </div>
            </div>
          </div>

          <div className="grid g2" style={{ marginTop: 18 }}>
            <div className="panel">
              <Swipe before={scene.images.before} after={scene.images.after} />
              <div className="fig-cap">
                Source in blue, reference in red. Colour fringing is misregistration;
                neutral grey is alignment.
              </div>
            </div>
            <div className="panel">
              <TiePointMap scene={scene} />
              <div className="fig-cap">
                Tie points over the uniformity lattice, coloured by residual.
              </div>
            </div>
          </div>

          <div className="grid g2" style={{ marginTop: 18 }}>
            <figure className="panel fig wide">
              <img src={url(scene.images.checkerboard)} alt="Checkerboard comparison" />
              <figcaption>Checkerboard against the reference &mdash; edges should run straight across the seams</figcaption>
            </figure>
            <figure className="panel fig wide">
              <img src={url(scene.images.shadow)} alt="Ray-cast shadow mask" />
              <figcaption>Ray-cast shadow mask used by the render (horizon sweep, S2a)</figcaption>
            </figure>
          </div>

          <div className="note" style={{ marginTop: 18 }}>
            <strong>Ingestible as delivered.</strong> Each run writes a registered
            cloud-optimised GeoTIFF, a tie-point list carrying a 2&times;2 covariance per
            point in CSV and GeoJSON, the transform, a PDS4-style label and a
            self-contained QA report.
          </div>
        </div>
      </div>
    </section>
  )
}
