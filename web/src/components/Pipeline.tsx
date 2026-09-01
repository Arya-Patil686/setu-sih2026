import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import { gsap, ScrollTrigger } from '../lib/motion'

interface Stage {
  id: string
  title: string
  note: string
  novelty?: string
  accent?: boolean
}

const STAGES: Stage[] = [
  { id: 'S0', title: 'Ingest', note: 'PDS4, PDS3 and GeoTIFF behind one Product type. Illumination resolved from backplane, then SPICE, then label keywords. Never guessed.' },
  { id: 'S1', title: 'Geometric pre-alignment', note: 'Both images ortho-projected onto one map projection at one GSD, on SLDEM2015. This is what collapses scale and viewpoint.', accent: true },
  { id: 'S2', title: 'Illumination harmonisation', note: 'The reference is re-rendered under the source image’s own solar geometry, with ray-cast shadows.', novelty: 'N1' },
  { id: 'S3', title: 'Correspondence, two tracks', note: 'A pretrained cross-modality network and a phase-congruency structural matcher, run independently and gated on agreement.', novelty: 'N2' },
  { id: 'S4', title: 'Sub-pixel refinement', note: 'Upsampled phase correlation and least-squares matching, each point emerging with a 2×2 covariance.', novelty: 'N4' },
  { id: 'S5', title: 'Outlier rejection and model', note: 'MAGSAC++ with a threshold derived from the covariances, then a cross-validated local model and a per-row jitter spline.' },
  { id: 'S6', title: 'Uniformity enforcement', note: 'A lattice quota over the true overlap, farthest-point selection inside each cell, and empty cells re-seeded against the known model.', novelty: 'N3' },
  { id: 'S7', title: 'Product generation', note: 'Registered GeoTIFF, tie points as CSV and GeoJSON, a PDS4-style label, and a self-contained QA report.' },
  { id: 'S8', title: 'Evaluation harness', note: 'Built before the matcher, on purpose. Every claim on this page is a number this module produced.', novelty: 'N5', accent: true },
]

const PLATE_W = 300
const PLATE_H = 74
const SKEW = 46

export default function Pipeline() {
  const scope = useRef<HTMLElement>(null)

  useGSAP(
    () => {
      const mm = gsap.matchMedia()

      mm.add('(min-width: 901px) and (prefers-reduced-motion: no-preference)', () => {
        // The stack arrives collapsed and separates as the section passes through the
        // viewport, the way an exploded assembly drawing pulls apart.
        //
        // Deliberately not pinned. Pinning would hold the diagram still while it
        // separates, which reads slightly better, but a pinned section that fails to
        // release traps the reader on the page with no way forward. On a submission that
        // will be opened on hardware and browsers we cannot test, the version that cannot
        // strand anybody is the right one, and scrubbing the section's own progress gets
        // most of the effect anyway.
        const tl = gsap.timeline({
          scrollTrigger: {
            trigger: '.pipe-stage',
            start: 'top bottom-=80',
            end: 'bottom top+=180',
            scrub: 0.7,
          },
        })

        STAGES.forEach((_, i) => {
          tl.to(
            `.plate-${i}`,
            { y: i * (PLATE_H + 30) - (STAGES.length - 1) * (PLATE_H + 30) * 0.5, ease: 'none' },
            0,
          )
          tl.to(`.lead-${i}`, { opacity: 1, ease: 'none' }, 0.15)
          tl.to(`.anno-${i}`, { opacity: 1, x: 0, ease: 'none' }, 0.15 + i * 0.02)
        })
      })

      mm.add('(max-width: 900px), (prefers-reduced-motion: reduce)', () => {
        // No pinning, no explosion: the stages become a plain list, which is the right
        // shape on a phone and the right behaviour for anyone who asked for less motion.
        STAGES.forEach((_, i) => {
          gsap.set(`.plate-${i}`, { y: i * (PLATE_H + 30) - (STAGES.length - 1) * (PLATE_H + 30) * 0.5 })
          gsap.set(`.lead-${i}`, { opacity: 1 })
          gsap.set(`.anno-${i}`, { opacity: 1, x: 0 })
        })
        ScrollTrigger.refresh()
      })
    },
    { scope },
  )

  return (
    <section id="pipeline" ref={scope}>
      <div className="wrap">
        <p className="eyebrow">The system</p>
        <h2>Nine stages, two feedback edges</h2>
        <p className="lede">
          The order is not decorative. The evaluation harness is built before the matcher,
          because the entire design rests on measured comparison rather than on a
          screenshot. Two edges run backwards: once a global transform exists, the
          footprint is re-projected from it, and cells that fail their quota are re-seeded
          against a model that is now known.
        </p>
      </div>

      <div className="pipe-stage">
        <div className="wrap pipe-wrap">
          <div className="pipe-diagram" aria-hidden="true">
            {STAGES.map((s, i) => (
              <div className={`plate plate-${i}${s.accent ? ' accent' : ''}`} key={s.id}>
                <svg width={PLATE_W + SKEW} height={PLATE_H + 30} viewBox={`0 0 ${PLATE_W + SKEW} ${PLATE_H + 30}`}>
                  <defs>
                    <linearGradient id={`pg-${i}`} x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor={s.accent ? '#123946' : '#101827'} />
                      <stop offset="100%" stopColor={s.accent ? '#0a2530' : '#0a0f1a'} />
                    </linearGradient>
                  </defs>
                  {/* Isometric plate: a parallelogram with a thin extruded edge. */}
                  <path
                    d={`M ${SKEW} 0 L ${PLATE_W + SKEW} 0 L ${PLATE_W} ${PLATE_H} L 0 ${PLATE_H} Z`}
                    fill={`url(#pg-${i})`}
                    stroke={s.accent ? '#22d3ee' : '#243149'}
                    strokeWidth="1"
                  />
                  <path
                    d={`M 0 ${PLATE_H} L ${PLATE_W} ${PLATE_H} L ${PLATE_W} ${PLATE_H + 9} L 0 ${PLATE_H + 9} Z`}
                    fill={s.accent ? '#0a1e26' : '#070b12'}
                    stroke={s.accent ? 'rgba(34,211,238,.45)' : '#1a2333'}
                    strokeWidth="1"
                  />
                  <text x={SKEW + 20} y={30} className="plate-id" fill={s.accent ? '#22d3ee' : '#5d6d85'}>
                    {s.id}
                  </text>
                  <text x={SKEW + 20} y={52} className="plate-title" fill="#e8eef7">
                    {s.title}
                  </text>
                </svg>
              </div>
            ))}

            {/* Leader lines, drawn from each plate out to its annotation. */}
            <svg className="leaders" viewBox="0 0 1100 760" preserveAspectRatio="none">
              {STAGES.map((_, i) => {
                const y = 380 + (i - (STAGES.length - 1) / 2) * (PLATE_H + 30) + 26
                const right = i % 2 === 1
                return (
                  <g className={`lead lead-${i}`} key={i}>
                    <path
                      d={right ? `M 700 ${y} L 760 ${y} L 800 ${y}` : `M 330 ${y} L 270 ${y} L 230 ${y}`}
                      stroke="#22d3ee"
                      strokeWidth="1"
                      fill="none"
                      opacity="0.5"
                    />
                    <circle cx={right ? 700 : 330} cy={y} r="2.5" fill="#22d3ee" />
                  </g>
                )
              })}
            </svg>

            {STAGES.map((s, i) => (
              <div
                className={`anno anno-${i} ${i % 2 === 1 ? 'right' : 'left'}`}
                key={`a-${s.id}`}
                style={{
                  top: `calc(50% + ${(i - (STAGES.length - 1) / 2) * (PLATE_H + 30) + 26}px)`,
                }}
              >
                <div className="anno-head">
                  <span className="anno-id">{s.id}</span>
                  {s.novelty && <span className="anno-nov">{s.novelty}</span>}
                </div>
                <div className="anno-title">{s.title}</div>
                <p className="anno-note">{s.note}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="wrap">
        <div className="grid g2" style={{ marginTop: 40 }}>
          <div className="note">
            <strong>S5 &rarr; S1.</strong> Once a global transform is estimated, pre-alignment
            is re-run from the corrected footprint. One iteration is enough; two is the cap.
          </div>
          <div className="note">
            <strong>S6 &rarr; S3.</strong> Cells that fail their quota are re-searched with a
            lowered threshold and a &plusmn;5&nbsp;px window, because the model is now known.
            This is what actually delivers uniformity.
          </div>
        </div>
      </div>
    </section>
  )
}
