# FusAlign

Identify head/tail breakpoint regions in fusion oncoproteins, color aligned
segments in an HTML report, and map parent ↔ fusion residues after a run.

## Environment setup

```bash
conda create -n fusalign python=3.10
conda activate fusalign
python -m pip install -r requirements.txt
```

## Layout

```
FusAlign/
├── app.py                          # Gradio UI
├── fusion_breakpoint_pipeline.py   # Alignment + report + residue query
├── requirements.txt
├── examples/
└── README.md
```

## Run the Gradio app

```bash
python app.py
```

Paste a single fusion (name + fusion/head/tail sequences) or upload a CSV with
columns `fusion_name`, `fusion`, `head`, `tail`. Click **Generate report** to
produce a summary table, HTML preview, and downloadable HTML/CSV.

Display colors use color pickers:

- head color
- tail color
- mutation color

If you only change colors and click **Generate report** again with the same
biological input, FusAlign reuses the previous alignment and applies a fast
color-only refresh.

## Sequence coloring

In colored sequences (report sections and summary thumbnails):

- **Head** residues: crimson (or your chosen head color)
- **Tail** residues: royalblue (or your chosen tail color)
- **Mutations** (parent AA ≠ fusion AA at an aligned site): purple (`#7c3aed`) and underlined

A short legend appears above each colored-sequence block and in the summary preview.

## Residue query

After a successful analysis you can map residues **without re-running**
Smith–Waterman. Queries use the **parent sequences you provided** (not UniProt
isoform remapping).

### Syntax

| Query | Meaning |
| --- | --- |
| `Tail:Y1078` | Tail parent position 1078; check that parent (and report fusion) AA is Y |
| `TAIL: Y1078` | Same, case/spacing flexible |
| `Tail: Y 1078` | Same, spacing flexible |
| `Tail:1078` | Position only (no AA check) |
| `Tail:1078-1085` | Range query (inclusive) |
| `Head:S12` | Head parent residue |
| `Fusion:Y200` | Fusion residue → mapped Head/Tail parent site |
| `ALK:Y1078` | Alias query (tail partner name) when fusion name is `Head::Tail` or `Head-Tail` |

Partner names are case-insensitive: `Head` / `Tail` / `Fusion`.
If the fusion name has an explicit separator (`::` or `-`), the partner names
are accepted as aliases (for example, `EML4::ALK` enables `EML4:*` and `ALK:*`).

### Answers

- **Found**: parent ↔ fusion indices and amino acids; whether the fusion still
  carries the requested letter; note if the aligned pair is a substitution
- **Not in alignment**: residue outside the head/tail hit span (span reported)
- **Parent AA disagreement**: position exists but the provided parent sequence
  has a different letter than requested (e.g. asked `Y1078` but parent has `F`)

### Where to query

**Gradio**: after **Generate report**, use the **Residue query** panel
(fusion dropdown + textbox).

## Python API (brief)

```python
from fusion_breakpoint_pipeline import (
    run_fusion_breakpoint_batch,
    query_residue,
    format_residue_query_answer,
)
import json
import pandas as pd

df = pd.read_csv("examples/example_fusion_breakpoint_input.csv")
results, html_path, csv_path = run_fusion_breakpoint_batch(df, out_dir="out")

maps = json.loads(results.loc[0, "residue_maps_json"])
print(format_residue_query_answer(query_residue(maps, query_text="Head:1")))
```
