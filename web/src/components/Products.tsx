import { useRef } from 'react'
import { useReveal } from '../lib/reveal'

const FILES = [
  ['registered.tif', 'Source resampled into the reference geometry. Cloud-optimised GeoTIFF, Lanczos, full CRS and affine, nodata set.'],
  ['tiepoints.csv', 'One row per correspondence: pixel and selenographic coordinates, confidence, track, σx σy σxy, residuals, inlier and re-seed flags, lattice cell.'],
  ['tiepoints.geojson', 'The same list as a FeatureCollection, ready to open in QGIS.'],
  ['transform.json', 'Global model matrix, local model coefficients, the per-row jitter spline, CRS, GSD and the resolved configuration.'],
  ['metrics.json', 'Every metric in the evaluation protocol, plus the per-stage timing of the run that produced them.'],
  ['label.xml', 'PDS4-style label: Identification_Area, Observation_Area carrying both product IDs and the illumination state, File_Area_Observational.'],
  ['report.html', 'Self-contained QA report with overlays, residual field, uniformity map and the metric table, with every image embedded.'],
  ['blink.gif', 'Animated before-and-after comparison, for people who read pictures faster than tables.'],
]

const CLI = [
  ['setu register', '--source ch2_ohr_… .xml --reference M1234567890LE.IMG --dem sldem_tile.tif --out runs/01'],
  ['setu bench generate', '--sun-elev 10,20,30,45,60,75 --sun-az 0,45,…,315 --scale 1,2,4,8,16 --out data/bench'],
  ['setu eval', '--methods all --out runs/eval_full'],
  ['setu serve', '--port 8000'],
]

export default function Products() {
  const scope = useRef<HTMLElement>(null)
  useReveal('.p-row')

  return (
    <section id="products" ref={scope}>
      <div className="wrap">
        <p className="eyebrow">What comes out</p>
        <h2 style={{ maxWidth: '20ch' }}>A registered product, a tie-point list with covariance, and a label</h2>
        <p className="lede">
          The problem statement asks for three artefacts, not one: the software, the
          registered image with its match-point list, and a quantitative evaluation. Every
          run writes all three into a single directory.
        </p>

        <div className="panel table-scroll p-table" style={{ marginTop: 34 }}>
          <table>
            <thead>
              <tr><th>File</th><th>Contents</th></tr>
            </thead>
            <tbody>
              {FILES.map(([f, d]) => (
                <tr className="p-row" key={f}>
                  <td className="mono" style={{ whiteSpace: 'nowrap', color: 'var(--accent)' }}>{f}</td>
                  <td style={{ color: 'var(--muted)' }}>{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: 44 }}>
          <p className="eyebrow">Command line</p>
          <div className="panel cli">
            {CLI.map(([cmd, args]) => (
              <div className="cli-row" key={cmd}>
                <span className="cli-cmd mono">{cmd}</span>
                <span className="cli-args mono">{args}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="grid g3" style={{ marginTop: 30 }}>
          <div className="note">
            <strong>Runs on a laptop.</strong> The classical track needs only NumPy, SciPy
            and OpenCV. The deep track uses whatever is available and reports which
            checkpoint produced each result rather than assuming one.
          </div>
          <div className="note">
            <strong>Nothing hard-codes a payload.</strong> OHRC, TMC-2, IIRS, LRO NAC,
            Kaguya TC and WAC all enter through one <span className="mono">Product</span>{' '}
            abstraction, and the reference policy is a table you can read.
          </div>
          <div className="note">
            <strong>It refuses to guess.</strong> If the solar geometry cannot be resolved
            from a backplane, from SPICE, or from label keywords, the run stops. An invented
            sun angle would poison every number downstream of it.
          </div>
        </div>
      </div>
    </section>
  )
}
