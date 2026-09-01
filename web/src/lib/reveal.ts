import { useEffect } from 'react'
import { ScrollTrigger, prefersReducedMotion } from './motion'

/** Re-measure the scrubbed timelines when a hidden page becomes visible again. */
function ScrollTriggerSafeRefresh() {
  try {
    ScrollTrigger.refresh()
  } catch {
    /* nothing to refresh */
  }
}

/**
 * Entrance reveals driven by IntersectionObserver rather than scroll position.
 *
 * ScrollTrigger computes each element's start offset once and needs re-measuring whenever
 * the document height changes. On a page that loads thirty images from the demo bundle
 * that happens constantly, and a stale offset means content that never appears. The
 * failure is silent and it looks like a broken page.
 *
 * IntersectionObserver asks the browser whether the element is actually on screen, so
 * there is nothing to invalidate. GSAP still drives the hero and the scrubbed pipeline
 * diagram, where a timeline is genuinely what is wanted.
 */
export function useReveal(selector = '.reveal, .d-reveal, .r-reveal, .p-row', deps: unknown[] = []) {
  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll<HTMLElement>(selector))
    if (nodes.length === 0) return

    if (prefersReducedMotion() || typeof IntersectionObserver === 'undefined') {
      nodes.forEach((el) => el.classList.add('is-in'))
      return
    }

    nodes.forEach((el, i) => {
      el.classList.add('will-reveal')
      // A small stagger inside a group, capped so a long list never crawls.
      el.style.setProperty('--reveal-delay', `${Math.min(i % 6, 5) * 55}ms`)
    })

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('is-in')
            io.unobserve(e.target)      // reveal once; scrolling back must not hide it
          }
        })
      },
      { rootMargin: '0px 0px -12% 0px', threshold: 0.02 },
    )

    nodes.forEach((el) => io.observe(el))

    // Reveal anything already on screen synchronously, without waiting for the observer
    // to deliver its first callback. IntersectionObserver only fires while the page is
    // being painted, so in a background tab or a hidden frame the first batch can arrive
    // arbitrarily late, and geometry read directly does not have that problem.
    const showIfOnScreen = () => {
      nodes.forEach((el) => {
        const r = el.getBoundingClientRect()
        if (r.top < window.innerHeight && r.bottom > 0) {
          el.classList.add('is-in')
          io.unobserve(el)
        }
      })
    }
    showIfOnScreen()
    window.addEventListener('scroll', showIfOnScreen, { passive: true })
    window.addEventListener('resize', showIfOnScreen, { passive: true })

    // A hard guarantee that nothing stays invisible. IntersectionObserver does not fire
    // while a page is not being painted (a background tab, a hidden frame, some embedded
    // viewers), and an entrance animation that fails closed leaves a reader looking at a
    // blank section with no way to recover. After a few seconds anything still hidden is
    // simply shown: losing the animation is a far better failure than losing the content.
    const failSafe = window.setTimeout(() => {
      nodes.forEach((el) => el.classList.add('is-in'))
    }, 2600)

    const onVisible = () => {
      if (document.visibilityState === 'visible') ScrollTriggerSafeRefresh()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      io.disconnect()
      window.clearTimeout(failSafe)
      window.removeEventListener('scroll', showIfOnScreen)
      window.removeEventListener('resize', showIfOnScreen)
      document.removeEventListener('visibilitychange', onVisible)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}
