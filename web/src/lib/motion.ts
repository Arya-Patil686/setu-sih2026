import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Lenis from 'lenis'

gsap.registerPlugin(ScrollTrigger)

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

/**
 * Smooth scrolling driven by GSAP's ticker rather than its own loop.
 *
 * Lenis and ScrollTrigger each want a requestAnimationFrame of their own, and left
 * alone the two drift out of phase — scroll-linked animation then jitters in a way that
 * reads as a performance problem and is not one. One clock, and Lenis is told about it:
 * `autoRaf: false`, and the ticker's seconds converted to the milliseconds Lenis expects.
 */
export function startSmoothScroll(): Lenis | null {
  if (typeof window === 'undefined') return null
  if (prefersReducedMotion()) return null

  const lenis = new Lenis({
    autoRaf: false,
    duration: 1.05,
    wheelMultiplier: 0.9,
  })

  lenis.on('scroll', ScrollTrigger.update)
  gsap.ticker.add((time) => lenis.raf(time * 1000))
  gsap.ticker.lagSmoothing(0)

  return lenis
}

/** Split a sentence into per-word spans so each can be animated independently. */
export function splitWords(text: string): string[] {
  return text.split(/\s+/).filter(Boolean)
}

/**
 * Recompute every ScrollTrigger position once the page has actually settled.
 *
 * ScrollTrigger measures start and end offsets when a trigger is created. This page then
 * loads around thirty images from the demo bundle, every one of which changes the height
 * of the document below it, so by the time a reader arrives at a section its trigger is
 * pointing at a scroll position that no longer exists. The visible symptom is content
 * that never fades in, which looks like a broken page rather than a stale measurement.
 *
 * Refreshing on each image load is the reliable fix, with a couple of timed refreshes to
 * catch fonts and anything that settles late.
 */
export function refreshTriggersWhenSettled(): () => void {
  if (typeof window === 'undefined') return () => {}

  const refresh = () => ScrollTrigger.refresh()
  const timers = [400, 1200, 2500].map((ms) => window.setTimeout(refresh, ms))

  const images = Array.from(document.images)
  const pending = images.filter((img) => !img.complete)
  pending.forEach((img) => {
    img.addEventListener('load', refresh, { once: true })
    img.addEventListener('error', refresh, { once: true })
  })
  window.addEventListener('load', refresh)

  return () => {
    timers.forEach(window.clearTimeout)
    window.removeEventListener('load', refresh)
  }
}

export { gsap, ScrollTrigger }
