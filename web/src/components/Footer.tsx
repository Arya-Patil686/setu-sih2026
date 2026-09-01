const REFS = [
  ['MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training', 'He et al., TPAMI 2026 · arXiv:2501.07556'],
  ['RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform', 'Li, Hu & Ai, IEEE TIP 29:3296, 2020'],
  ['Robust registration of multimodal remote sensing images (HOPC) and CFOG', 'Ye et al., IEEE TGRS 55(5) 2017; ISPRS 2019'],
  ['Image Features from Phase Congruency', 'Kovesi, 1999'],
  ['MoonMetaSync: Lunar Image Registration Analysis', 'Kumar, Kaushal & Murthy, arXiv:2410.11118, 2024'],
  ['Geodetically Anchored 0.30 m DEM of the Chandrayaan-3 Vikram Landing Site', 'Tungathurthi, arXiv:2602.14993, 2026'],
  ['A new lunar digital elevation model from LOLA and SELENE Terrain Camera (SLDEM2015)', 'Barker et al., Icarus 273:346, 2016'],
  ['Chandrayaan-2 Orbiter High Resolution Camera: design, development and in-orbit performance', 'Chowdhury et al., Current Science 118(4):560, 2020'],
  ['Lunar surface temperature estimation and thermal emission correction using Chandrayaan-2 IIRS data', 'Verma, Chauhan & Chauhan, Icarus 383:115075, 2022'],
  ['MAGSAC++, a fast, reliable and accurate robust estimator', 'Barath et al., CVPR 2020'],
]

export default function Footer() {
  return (
    <footer>
      <div className="wrap">
        <div className="grid g3" style={{ gap: 40 }}>
          <div>
            <h4>SETU</h4>
            <p style={{ margin: 0, maxWidth: '34ch', lineHeight: 1.6 }}>
              Sub-pixel multi-sensor registration of Chandrayaan-2 imagery against lunar
              reference maps. Smart India Hackathon 2026, problem statement 26166,
              ISRO / Department of Space.
            </p>
            <p style={{ marginTop: 16, color: 'var(--muted)' }}>
              Geometry for scale and viewpoint. Physics for the Sun. Learning only for what
              is left.
            </p>
          </div>
          <div>
            <h4>Data sources</h4>
            <ul>
              <li>Chandrayaan-2 L1 via PRADAN (OHRC, TMC-2, IIRS)</li>
              <li>LRO NAC and NAC DTMs via LROC / ODE</li>
              <li>SLDEM2015, 512 ppd</li>
              <li>Kaguya TC ortho and DEM via JAXA DARTS</li>
              <li>LROC WAC global mosaic</li>
            </ul>
          </div>
          <div>
            <h4>Key references</h4>
            <ul>
              {REFS.slice(0, 6).map(([t, c]) => (
                <li key={t}>
                  <span style={{ color: 'var(--muted)' }}>{t}</span>
                  <br />
                  <span style={{ fontSize: 12 }}>{c}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <p style={{ marginTop: 44, paddingTop: 22, borderTop: '1px solid var(--line)' }}>
          Every figure on this page was produced by the evaluation harness in this
          repository. The benchmark&rsquo;s ground truth is exact by construction: both images
          of every pair are rendered from one terrain model under a transform that is known
          rather than estimated.
        </p>
      </div>
    </footer>
  )
}
