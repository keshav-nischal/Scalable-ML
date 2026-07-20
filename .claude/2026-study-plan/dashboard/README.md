# Dashboard

An interactive control panel for the study plan — generated from the markdown in
`../` (the plan is the source of truth; this is the operable view on top).

## File layout

The dashboard is split into four files so each part stays small and editable
(open only the one you need instead of one ~1200-line file):

| File             | What's in it                                                   |
| ---------------- | -------------------------------------------------------------- |
| `dashboard.html` | Thin shell — page markup, loads the CSS + JS below             |
| `data.js`        | **The plan data** (weeks, blocks, projects, posts, checkpoints) — edit this when the plan changes |
| `app.js`         | Rendering + progress engine + file sync (machinery; rarely changes) |
| `styles.css`     | All styling / theme tokens                                     |

`data.js` defines the globals `WEEKS`, `BLOCKS`, `PROJECTS`, `REPROS`, `POSTS`,
`CPS` (plus the `TYPE`/`it()` helpers) and loads **before** `app.js`, which reads
them. Loaded as classic scripts + a stylesheet `<link>`, so it works opened
directly (`file://`) or served. **To change plan content, edit `data.js` only.**

## Use it

Open **`dashboard.html`** in a browser (double-click, or serve the folder — see below).
It auto-selects the current week from today's date, tracks progress with checkboxes,
and shows timeline / block / project / post completion.

## Progress storage

- Progress always saves **instantly in the browser** (localStorage).
- To keep it as a real, commit-able file, use the **Save & sync** panel:
  - **Chrome / Edge:** click **Save to file** once and pick `progress.json` in *this*
    folder. After that it **auto-saves** to that file on every change, and on
    subsequent visits it **re-attaches and loads itself automatically** — no clicks
    (if the browser still remembers the permission; otherwise a one-tap
    **Reconnect** appears). Commit `progress.json` to version your progress.
  - **Other browsers / restricted `file://`:** **Save to file** downloads
    `progress.json` (move it here) and **Load from file** re-imports it.
- **Auto-detection:** when the folder is served over `http://localhost`, a committed
  `progress.json` sitting next to `dashboard.html` is **detected and loaded
  automatically** on open (read-only until you click **Save to file** once to enable
  writing). The newer of {file, this browser} wins, compared by `savedAt`, so a
  freshly pulled `progress.json` is picked up without clobbering newer local edits.
- Guaranteed auto-save-in-place: serve the folder over localhost so the browser's
  file API is fully enabled, e.g. from this directory:

  ```
  python -m http.server 8000
  ```

  then open `http://localhost:8000/dashboard.html`.

## Regenerating

Ask Claude Code to update the dashboard whenever the plan changes — in practice that
means editing `data.js` (and `../*.md`) to match. `app.js`/`styles.css` only change
when the layout or styling changes. `progress.json` is independent and is not overwritten.
