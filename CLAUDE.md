# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Git

- Author all commits as **keshav-nischal &lt;keshavnischal@gmail.com&gt;**. The local
  git config is already set to this — keep it; never commit under another identity.

## Scope

- In this repo, help only with **committing and pushing changes**, with
  maintaining my planning notes under `.claude/2026-study-plan/`, and with
  **drafting LinkedIn posts** (see below).
- The exercises, notebooks, and code under `src/` are **my own work** — I solve
  them by hand as part of studying. Don't write, complete, or modify them.

## LinkedIn post drafts — write them in *my* format

Drafts live in `.claude/2026-study-plan/posts/`. `brand.md` covers strategy (what's
worth posting); this covers **form**. Both of my real posts follow the same shape —
match it, don't improve on it:

1. **One opening line:** what I built, plus a dash clause naming the constraint.
   *"Built a neural network in NumPy to solve XOR — no frameworks, just matrices and
   backprop."* / *"Re-implemented the same neural network in PyTorch — this time with
   the framework doing the backprop."*
2. **A blank line after it**, then the body.
3. **One sentence per line. No paragraphs.** Short, flat declaratives.
4. **Name what helped** — the book, the chapter, the video (3Blue1Brown, "Hands-On ML").
5. **Say what broke, plainly.** Both posts admit the same shape-mismatch mess. Never
   sand this off.
6. **End on what changed in my understanding**, not on an achievement.
7. **`Next step:`** — one line, what I'm doing next. It chains the posts together.
8. **`🔗 Code:`** then the link, on the next line.

Keep it roughly as short as post #1. Technical terms go in bare (`nn.Module`,
`[N, 1]`) — no explaining them.

**Things I've rejected drafts for — don't do these:**

- Essay prose: long em-dash sentences, paragraphs, a build to a punchline.
- Polished aphorisms or closing zingers. Anything quotable reads as fake.
- Code blocks, bold, headers, bullet lists, rhetorical questions to the reader.
- "The thing I'll actually remember...", "Here's what surprised me...", any framing
  that announces its own insight.
- Hashtags, emojis beyond the 🔗, CTAs, "excited to share".
- Inventing detail. Every number, bug, and file has to come from what I actually ran —
  if I can't point at the cell it came from, cut the line. Flag anything you're unsure
  of as a pre-post check instead of smoothing over it.

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
