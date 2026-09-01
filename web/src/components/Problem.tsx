import { useEffect, useRef } from 'react'
import { gsap, prefersReducedMotion } from '../lib/motion'
import { useReveal } from '../lib/reveal'

interface Props {
  illum?: {
    images: Record<string, string>
    ncc_opposite_azimuth: number
    ncc_elevation_change: number
    ncc_structural_opposite: number
    shadow_fraction_low_sun: number
    shadow_fraction_high_sun: number
  }
}

const base = import.meta.env.BASE_URL || '/'
const url = (p?: string) => (p ? `${base}${p}`.replace(/\/{2,}/g, '/') : undefined)

export default function Problem({ illum }: Props) {
  const scope = useRef<HTMLElement>(null)

  useReveal('.reveal', [illum])

  // The number counts down into negative territory, which is the point: it is not that
  // the correlation weakens between the two images, it is that it inverts. Fired from an
  // observer for the same reason the reveals are: a scroll offset measured before the
  // images load is a scroll offset that never arrives.
  useEffect(() => {
    const el = scope.current?.querySelector<HTMLElement>('.ncc-big .v')
    if (!el) return
    const value = illum?.ncc_opposite_azimuth ?? -0.99

    if (prefersReducedMotion()) {
      el.textContent = value.toFixed(3)
      return
    }

    let done = false
    const run = () => {
      if (done) return
      done = true
      const target = { v: 1 }
      gsap.to(target, {
        v: value,
        duration: 1.6,
        ease: 'power3.out',
        onUpdate: () => { el.textContent = target.v.toFixed(3) },
      })
    }

    const check = () => {
      const r = el.getBoundingClientRect()
      if (r.top < window.innerHeight * 0.85 && r.bottom > 0) run()
    }

    check()
    window.addEventListener('scroll', check, { passive: true })
    // If the animation never gets its chance, the number still has to be on the page.
    const failSafe = window.setTimeout(() => {
      if (!done) { done = true; el.textContent = value.toFixed(3) }
    }, 4000)

    return () => {
      window.removeEventListener('scroll', check)
      window.clearTimeout(failSafe)
    }
  }, [illum])

  return (
    <section id="problem" ref={scope}>
      <div className="wrap">
        <p className="eyebrow reveal">Why this is hard</p>
        <h2 className="reveal" style={{ maxWidth: '19ch' }}>
          On the Moon, a crater lit from the east is the photographic negative of the same
          crater lit from the west.
        </h2>
        <p className="lede reveal">
          There is no atmosphere to soften a shadow and almost no albedo variation to key
          on, so appearance is a property of the Sun rather than of the surface. Both
          images below are the identical patch of ground, rendered from one terrain model
          at one viewpoint. Only the solar azimuth differs.
        </p>

        <div className="grid g3 reveal" style={{ marginTop: 44, alignItems: 'start' }}>
          <figure className="panel fig">
            <img src={url(illum?.images.sun_east)} alt="Crater field lit from the east at 18 degrees solar elevation" />
            <figcaption>Sun from the east · elevation 18&deg;</figcaption>
          </figure>
          <figure className="panel fig">
            <img src={url(illum?.images.sun_west)} alt="The same crater field lit from the west" />
            <figcaption>Sun from the west · elevation 18&deg;</figcaption>
          </figure>
          <div className="stat panel ncc-big bad" style={{ alignSelf: 'stretch' }}>
            <div className="k">Correlation between them</div>
            <div className="v">—</div>
            <div className="u">
              Not weak. <strong style={{ color: 'var(--fg)' }}>Inverted.</strong> Every
              gradient a descriptor keys on has changed sign, which is why SIFT, ORB and a
              MegaDepth-trained network all fail here rather than degrade.
            </div>
          </div>
        </div>

        <div className="grid g2 reveal" style={{ marginTop: 20 }}>
          <div className="note">
            <strong>The obvious fix does not work either.</strong> A published comparison of
            SIFT, ORB and a lunar-specific hybrid on real OHRC-to-TMC-2 pairs
            (MoonMetaSync, arXiv 2410.11118) reports SSIM around 0.75&ndash;0.79 and finds
            that low sun degrades every method, with the authors&rsquo; own hybrid failing to
            beat plain SIFT. That is the floor this project set out to beat.
          </div>
          <div className="note">
            <strong>And the prior is kilometres out.</strong> Chandrayaan-2 products are
            seleno-tagged, so the approximate footprint is known before any pixel is
            matched &mdash; but independent reprocessing of OHRC found SPICE-projected
            positions roughly 4&nbsp;km from truth. A kilometre-level prior against a
            sub-pixel requirement is the whole problem.
          </div>
        </div>

        <div className="reveal" style={{ marginTop: 56 }}>
          <p className="eyebrow">The move</p>
          <h3 style={{ fontSize: 26, letterSpacing: '-0.02em', maxWidth: '26ch' }}>
            Stop looking for a descriptor that survives the Sun. Remove the Sun from the
            problem instead.
          </h3>
          <p className="lede" style={{ marginTop: 16 }}>
            Two of the three variations the problem statement names &mdash; scale and
            viewpoint &mdash; are geometry, and Chandrayaan-2 ships the geometry in every
            product. Spend it. The third is genuine appearance, so the reference is
            re-rendered from the terrain model under the source image&rsquo;s own solar
            azimuth, elevation and emission angle. Matching then happens between two images
            that already agree about where the Sun is.
          </p>
        </div>
      </div>
    </section>
  )
}
