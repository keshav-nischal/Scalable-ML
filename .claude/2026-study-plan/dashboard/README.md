# Dashboard

An interactive control panel for the study plan — generated from the markdown in
`../` (the plan is the source of truth; this is the operable view on top).

## Use it

Open **`dashboard.html`** in a browser (double-click, or serve the folder — see below).
It auto-selects the current week from today's date, tracks progress with checkboxes,
and shows timeline / block / project / post completion.

## Progress storage

- Progress always saves **instantly in the browser** (localStorage).
- To keep it as a real, commit-able file, use the **Save & sync** panel:
  - **Chrome / Edge:** click **Save to file** once and pick `progress.json` in *this*
    folder. After that it **auto-saves** to that file on every change. Commit
    `progress.json` to version your progress.
  - **Other browsers / restricted `file://`:** **Save to file** downloads
    `progress.json` (move it here) and **Load from file** re-imports it.
- Guaranteed auto-save-in-place: serve the folder over localhost so the browser's
  file API is fully enabled, e.g. from this directory:

  ```
  python -m http.server 8000
  ```

  then open `http://localhost:8000/dashboard.html`.

## Regenerating

Ask Claude Code to regenerate `dashboard.html` whenever the plan changes — it rebuilds
from the markdown files in `../`. `progress.json` is independent and is not overwritten.
