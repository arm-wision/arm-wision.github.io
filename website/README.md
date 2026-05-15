# PlantCLEF 2026 — Website (Next.js 14)

Next.js 14 App Router app for the team's PlantCLEF 2026 submission recap.
Static export, no backend. Hostable on GitHub Pages, Cloudflare Pages, Vercel,
or any static file server.

The sibling folder `../plantclef/` is the original static-React design
reference (Babel-standalone, no build step) — kept for diffing the
original design while we evolve this app.

## Quick start

```bash
cd website/plantclef-next
npm install
npm run dev          # localhost:3000
npm run build        # static export to ./out/
```

The build produces a fully static `out/` directory. Push it to a static
host, no Node runtime needed.

## Layout

```
plantclef-next/
├── app/
│   ├── layout.jsx          # html shell, Google Fonts
│   ├── page.jsx            # composes the section components
│   └── globals.css         # design tokens, layout, charts
├── components/
│   ├── Topbar.jsx          ("use client") — scroll-aware nav
│   ├── Hero.jsx
│   ├── Stats.jsx
│   ├── Abstract.jsx
│   ├── Tasks.jsx           ("use client") — tabbed task description
│   ├── Dataset.jsx
│   ├── ChartsSection.jsx   — wrapper for the five charts
│   ├── Timeline.jsx
│   ├── Leaderboard.jsx     ("use client") — sortable per-experiment table
│   ├── Baselines.jsx
│   ├── TeamBlock.jsx
│   ├── BibTeX.jsx          ("use client") — copy-to-clipboard
│   ├── Footer.jsx
│   └── charts/
│       ├── primitives.jsx        — YAxis, XAxis, Gridlines
│       ├── ScoreProgression.jsx  — Figure 1
│       ├── UnfreezeSweep.jsx     — Figure 2 (val/Kaggle inversion)
│       ├── ValVsKaggle.jsx       — Figure 3 (scatter + Pearson r)
│       ├── E015PerEpoch.jsx      — Figure 4
│       └── SpeciesLongTail.jsx   — Figure 5 (i003 cap histogram)
├── lib/
│   ├── team.js             — single source of truth for the team list
│   └── chart_data.js       — five chart datasets, regenerated from CSVs
└── public/
    └── data/               — JSON copies for ad-hoc fetches
        ├── experiments.json
        ├── submissions.json
        └── ...
```

## Where the data comes from

`website/plantclef/data/build_data.py` walks the project's local
`src_experiments/**/scores*.csv` files and the Kaggle CLI dump
(`kaggle competitions submissions plantclef-2026 -v`) and produces the seven
JSON files mirrored under `public/data/`. The chart components import the
same numbers verbatim from `lib/chart_data.js` so charts work even when
opened straight from the filesystem.

To refresh:

```bash
cd ../plantclef/data
kaggle competitions submissions plantclef-2026 -v --page-size 200 > kaggle_submissions.csv
PYTHONIOENCODING=utf-8 python build_data.py
cp *.json ../../plantclef-next/public/data/
# Then update the inline literals in plantclef-next/lib/chart_data.js
```

## Static export + hosting

`next.config.mjs` already has `output: 'export'` + `trailingSlash: true`.
After `npm run build`, the static site is in `./out/`. If you ever host
under a subpath (e.g. `*.github.io/PlantCLEF2026`), uncomment the
`basePath` line in `next.config.mjs` and rebuild.

## Migration history

This app was scaffolded from the design reference at `../plantclef/` per
`../MIGRATION_PLAN.md`. The migration:

1. Ported `sections.jsx` + `results.jsx` + `charts.jsx` into individual
   typed components under `components/`.
2. Replaced the placeholder leaderboard (12 fake competitor teams) with
   our real per-experiment best submissions.
3. Updated all dataset stats from the design-template placeholders
   (14,206 species → 7,806; 2,800 plots → 2,105; 3 tracks → 1 track).
4. Added five SVG charts derived from the project's actual training
   metrics and Kaggle submission scores.

`../MIGRATION_PLAN.md` documents what was done and what's still open.
