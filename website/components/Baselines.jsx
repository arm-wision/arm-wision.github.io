import Reveal from "./Reveal";

export default function Baselines() {
  const cards = [
    {
      tag: "OURS",
      title: "Method paper",
      copy: "Partial unfreeze sweet spot, long tail capping with extra under 100 data, and ablations across all 18 experiments.",
      meta: ["CEUR WS · PDF", "Forthcoming"],
      href: "#paper",
    },
    {
      tag: "OURS",
      title: "GitHub",
      copy: "End to end training scripts, manifest builders, DDP wrappers, per experiment summaries.",
      meta: ["github.com/ManindraDeMel/PlantCLEF2026", "PyTorch 2.x"],
      href: "https://github.com/ManindraDeMel/PlantCLEF2026",
    },
    {
      tag: "REFERENCE",
      title: "Experiments index",
      copy: "Per experiment summaries from 001 through i003 with recipe, val metrics, Kaggle scores, lessons.",
      meta: ["docs/experiments_summary.md", "18 experiments"],
      href: "https://github.com/ManindraDeMel/PlantCLEF2026/blob/main/docs/experiments_summary.md",
    },
  ];
  return (
    <section id="code">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Method and code</p>
            <h2>What we built.</h2>
          </div>
          <p className="lede">
            Paper, code, and per experiment writeups. Every numbered
            experiment under <code className="k-mono">src_experiments/</code>{" "}
            has its own <code className="k-mono">summary.md</code>.
          </p>
        </div>
        <Reveal className="cards" stagger>
          {cards.map((c) => (
            <a className="card" href={c.href} key={c.title}>
              <div className="tag">{c.tag}</div>
              <h4>{c.title}</h4>
              <p>{c.copy}</p>
              <div className="meta">
                <span>{c.meta[0]}</span>
                <span className="dot">·</span>
                <span>{c.meta[1]}</span>
              </div>
            </a>
          ))}
        </Reveal>
      </div>
    </section>
  );
}
