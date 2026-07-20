# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Git

- Author all commits as **keshav-nischal &lt;keshavnischal@gmail.com&gt;**. The local
  git config is already set to this — keep it; never commit under another identity.

## Scope

- In this repo, help only with **committing and pushing changes** and with
  maintaining my planning notes under `.claude/2026-study-plan/`.
- The exercises, notebooks, and code under `src/` are **my own work** — I solve
  them by hand as part of studying. Don't write, complete, or modify them.

## Study plan ↔ dashboard sync

- The plan notes (`.claude/2026-study-plan/*.md`) and the interactive dashboard
  (`.claude/2026-study-plan/dashboard/`) are both AI-maintained and describe the
  **same** plan. Keep them in sync: **any change to the plan must also update the
  dashboard, and vice versa** — never edit one and leave the other stale.
- The dashboard is split into `dashboard.html` (shell), `data.js` (the plan data),
  `app.js` (logic), and `styles.css`. **Plan content lives in `data.js`** — when
  changing week content, dates, blocks, projects, certs, checkpoints, posts, or
  numbering, apply the edit to both the relevant `.md` file(s) and the matching
  data in `data.js`. After editing, verify the two still agree before committing.
