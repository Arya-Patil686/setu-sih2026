import { useEffect, useRef } from 'react'
import { gsap, prefersReducedMotion } from '../lib/motion'

/**
 * The volumetric beam behind the hero.
 *
 * Deliberately CSS rather than WebGL. A shader would cost a GPU context and a render
 * loop to produce what two conic gradients and a blur already produce, and this page
 * spends its frame budget on the scroll-driven pipeline diagram instead. It also means
 * the effect degrades to a still image under reduced-motion with no separate code path.
 */
export default function LightCone({ intensity = 1 }: { intensity?: number }) {
  const scope = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (prefersReducedMotion() || !scope.current) return
    const ctx = gsap.context(() => {
      // A slow, shallow breath. Anything faster reads as a loading state.
      gsap.to('.cone', {
        opacity: 0.62,
        scaleX: 1.06,
        duration: 7,
        ease: 'sine.inOut',
        repeat: -1,
        yoyo: true,
      })
      gsap.to('.cone-core', {
        opacity: 0.72,
        duration: 4.5,
        ease: 'sine.inOut',
        repeat: -1,
        yoyo: true,
      })
    }, scope)
    return () => ctx.revert()
  }, [])

  return (
    <div className="cone-field" ref={scope} aria-hidden="true" style={{ opacity: intensity }}>
      <div className="limb" />
      <div className="cone" />
      <div className="cone-core" />
      <div className="grain" />
    </div>
  )
}
