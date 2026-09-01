import { useEffect, useState } from 'react'
import Topbar from './components/Topbar'
import Hero from './components/Hero'
import Problem from './components/Problem'
import Pipeline from './components/Pipeline'
import Demo from './components/Demo'
import Results from './components/Results'
import Products from './components/Products'
import Footer from './components/Footer'
import { loadDemo, loadEval, type DemoBundle, type EvalBundle } from './lib/data'
import { refreshTriggersWhenSettled } from './lib/motion'

export default function App() {
  const [demo, setDemo] = useState<DemoBundle | null>(null)
  const [evalData, setEvalData] = useState<EvalBundle | null>(null)

  useEffect(() => {
    loadDemo().then(setDemo)
    loadEval().then(setEvalData)
  }, [])

  // The demo images arrive after the triggers are created and change every offset below
  // them, so the positions have to be measured again once the page has settled.
  useEffect(() => refreshTriggersWhenSettled(), [demo, evalData])

  return (
    <div className="shell">
      <Topbar />
      <main>
        <Hero />
        <Problem illum={demo?.illumination} />
        <Pipeline />
        <Demo scenes={demo?.scenes ?? []} />
        <Results evalData={evalData} />
        <Products />
      </main>
      <Footer />
    </div>
  )
}
