import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import { gsap, splitWords } from '../lib/motion'
import LightCone from './LightCone'

const HEADLINE = 'Geometry for scale and viewpoint. Physics for the Sun. Learning only for what is left.'

export default function Hero() {
  const scope = useRef<HTMLElement>(null)

  useGSAP(
    () => {
      const mm = gsap.matchMedia()

      mm.add('(prefers-reduced-motion: no-preference)', () => {
        const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })
        tl.from('.hero-eyebrow', { opacity: 0, y: 14, duration: 0.5 })
          // Word by word, overlapping heavily. The sentence lights up as it is read,
          // which is the effect from the reference clips; a fully sequential version
          // reads as slow.
          .from(
            '.word > span',
            { yPercent: 108, opacity: 0, duration: 0.72, stagger: 0.035 },
            '-=0.25',
          )
          .from('.hero-sub', { opacity: 0, y: 18, duration: 0.55 }, '-=0.5')
          .from('.hero-actions > *', { opacity: 0, y: 14, duration: 0.45, stagger: 0.08 }, '-=0.35')
          .from('.badge', { opacity: 0, y: 10, duration: 0.4, stagger: 0.05 }, '-=0.3')
          .from('.scroll-cue', { opacity: 0, duration: 0.5 }, '-=0.2')
      })

      mm.add('(prefers-reduced-motion: reduce)', () => {
        gsap.set('.word > span, .hero-sub, .hero-actions > *, .badge, .hero-eyebrow, .scroll-cue', {
          opacity: 1,
          y: 0,
          yPercent: 0,
        })
      })
    },
    { scope },
  )

  return (
    <section className="hero" ref={scope}>
      <LightCone />
      <div className="wrap hero-inner">
        <p className="eyebrow hero-eyebrow" style={{ justifyContent: 'center' }}>
          Chandrayaan-2 · OHRC · TMC-2 · IIRS
        </p>

        <h1>
          {splitWords(HEADLINE).map((w, i) => (
            <span className="word" key={i}>
              <span>{w}&nbsp;</span>
            </span>
          ))}
        </h1>

        <p className="hero-sub">
          Sub-pixel correspondence between a Chandrayaan-2 optical image and a lunar
          reference map, robust to illumination, viewpoint and a hundredfold difference
          in scale, with a covariance on every tie point.
        </p>

        <div className="hero-actions">
          <a className="pill solid" href="#demo">See it run</a>
          <a className="pill" href="#results">Read the measurements</a>
        </div>

        <div className="badge-row">
          <span className="badge">Registered product</span>
          <span className="badge">Tie points + covariance</span>
          <span className="badge">PDS4-style label</span>
          <span className="badge">Runs on CPU</span>
        </div>
      </div>

      <div className="scroll-cue">
        <span>Scroll</span>
        <i />
      </div>
    </section>
  )
}
