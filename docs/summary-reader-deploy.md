# Summary Reader Deployment

The summary reader is a static site. It needs only the generated JSON exported
from `summaries_cache.json`; it does not need model API keys or vector database
access.

## Build locally

```powershell
.\scripts\export-summary-reader.ps1
```

```bash
./scripts/export-summary-reader.sh
```

The default output is `site\summary-reader`, which is ignored by git. To preview
it locally:

```powershell
.\scripts\export-summary-reader.ps1 -Serve -Port 8000
```

```bash
./scripts/export-summary-reader.sh --serve --port 8000
```

Open `http://localhost:8000`.

## Self-host

1. Run `.\scripts\export-summary-reader.ps1`.
2. Upload the contents of `site\summary-reader` to any static host.
3. Configure the host root to serve `index.html`.

The published folder must include:

- `index.html`
- `data\summaries.json`

## Zip production page

The production page from `Linkura Summaries.zip` is preserved separately under
`web\summary-reader-production`. Build it with:

```powershell
.\scripts\export-production-summary-reader.ps1
```

```bash
./scripts/export-production-summary-reader.sh
```

The default output is `site\summary-reader-production`. That generated folder
includes the zip page, `tweaks-panel.jsx`, and `data\summaries.json` using the
same normalized reader schema as the main static reader.

## GitHub Pages

Use one of these publishing patterns:

- Build to `site\summary-reader`, then publish that folder with a Pages-capable
  action or deployment tool.
- Build to `docs` when you intentionally want the generated reader data in git:

```powershell
.\scripts\export-summary-reader.ps1 -OutputDir docs
```

```bash
./scripts/export-summary-reader.sh --output-dir docs
```

Then set GitHub Pages to deploy from the `docs` folder. This is a publishing
choice; the repo does not require committing `summaries_cache.json` or generated
reader output for normal development.
