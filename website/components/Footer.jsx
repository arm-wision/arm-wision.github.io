export default function Footer() {
  return (
    <footer className="foot">
      <div className="foot-inner-recap">
        <div className="brand-block">
          <div className="brand-row">
            <span className="mark" aria-hidden="true" />
            <span>PlantCLEF&nbsp;2026 · Submission recap</span>
          </div>
          <p className="brand-tag">
            By Manindra de Mel, Arjun Raj, Razeen Wasif, Will Brake. Our
            submission to the 7th edition of the LifeCLEF plant
            identification challenge.
          </p>
        </div>
        <div className="foot-links">
          <a href="#paper">Paper</a>
          <a href="https://github.com/ManindraDeMel/PlantCLEF2026">GitHub</a>
          <a href="https://www.imageclef.org/PlantCLEF2026">Competition page ↗</a>
          <a href="#cite">Cite</a>
        </div>
      </div>
      <div className="foot-legal">
        <span>© 2026 de Mel, Raj, Wasif, Brake. ANU Deep Learning.</span>
        <span>Built for the LifeCLEF research evaluation initiative.</span>
      </div>
    </footer>
  );
}
