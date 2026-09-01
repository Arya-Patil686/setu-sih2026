import { useEffect, useState } from 'react'

const LINKS = [
  ['problem', 'Problem'],
  ['pipeline', 'Pipeline'],
  ['demo', 'Demo'],
  ['results', 'Results'],
  ['products', 'Products'],
]

export default function Topbar() {
  const [stuck, setStuck] = useState(false)

  useEffect(() => {
    const onScroll = () => setStuck(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header className={`topbar${stuck ? ' stuck' : ''}`}>
      <div className="brand">
        <span className="mark">SETU</span>
        <span className="sep">/</span>
        <span className="ctx">SIH 2026 · PS 26166</span>
      </div>
      <nav className="topnav">
        {LINKS.map(([id, label]) => (
          <a key={id} href={`#${id}`}>{label}</a>
        ))}
      </nav>
      <a className="pill" href="#demo">Run the demo</a>
    </header>
  )
}
