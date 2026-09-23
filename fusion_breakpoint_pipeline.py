# fusion_breakpoint_pipeline.py

import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from IPython.display import HTML as IPyHTML, display
except ImportError:
    IPyHTML = None
    display = None

@dataclass
class LocalHit:
    """A local alignment hit between a parent protein and the fusion sequence."""

    score: int
    fusion_start: int
    fusion_end: int
    protein_start: int
    protein_end: int
    aligned_pairs: Optional[list] = None


@dataclass
class FusionBreakpointResult:
    """Best inferred breakpoint result with head and tail alignment hits."""

    breakpoint_after: int
    score: int
    head_hit: LocalHit
    tail_hit: LocalHit


def traceback_aligned_pairs(dp, tb, protein, fusion, i, j):
    """Trace back aligned protein/fusion residue pairs from a Smith-Waterman matrix."""
    pairs = []

    while i > 0 and j > 0 and dp[i][j] > 0:
        direction = tb[i][j]

        if direction == "D":
            pairs.append({
                "protein_index_0ind": i - 1,
                "fusion_index_0ind": j - 1,
                "protein_aa": protein[i - 1],
                "fusion_aa": fusion[j - 1],
            })
            i -= 1
            j -= 1
        elif direction == "U":
            i -= 1
        elif direction == "L":
            j -= 1
        else:
            break

    return list(reversed(pairs))


def smith_waterman_hits_by_fusion_end(
    protein: str,
    fusion: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
):
    """
    Return the best local alignment hit ending at each fusion position.

    Coordinates are 0-based inclusive.
    """
    n = len(protein)
    m = len(fusion)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    start = [[None] * (m + 1) for _ in range(n + 1)]
    tb = [[None] * (m + 1) for _ in range(n + 1)]

    best_ending_at = [None] * m

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_score = match if protein[i - 1] == fusion[j - 1] else mismatch

            candidates = [
                (0, None, None),
                (dp[i - 1][j - 1] + sub_score, start[i - 1][j - 1], "D"),
                (dp[i - 1][j] + gap, start[i - 1][j], "U"),
                (dp[i][j - 1] + gap, start[i][j - 1], "L"),
            ]

            score, st, direction = max(candidates, key=lambda x: x[0])

            if score == 0:
                st = None
                direction = None
            elif st is None:
                st = (i - 1, j - 1)

            dp[i][j] = score
            start[i][j] = st
            tb[i][j] = direction

            if score > 0:
                protein_start, fusion_start = st

                aligned_pairs = traceback_aligned_pairs(
                    dp=dp,
                    tb=tb,
                    protein=protein,
                    fusion=fusion,
                    i=i,
                    j=j,
                )

                hit = LocalHit(
                    score=score,
                    fusion_start=fusion_start,
                    fusion_end=j - 1,
                    protein_start=protein_start,
                    protein_end=i - 1,
                    aligned_pairs=aligned_pairs,
                )

                old = best_ending_at[j - 1]
                if old is None or hit.score > old.score:
                    best_ending_at[j - 1] = hit

    return best_ending_at


def prefix_best_hits(hits):
    """Convert best hit ending exactly at j into best hit ending at or before j."""
    best = []
    current = None

    for hit in hits:
        if hit is not None and (current is None or hit.score > current.score):
            current = hit
        best.append(current)

    return best


def suffix_best_hits_tail(tail: str, fusion: str, **kwargs):
    """Find the best tail hit starting at or after each fusion position."""
    rev_tail = tail[::-1]
    rev_fusion = fusion[::-1]

    rev_hits_ending_at = smith_waterman_hits_by_fusion_end(
        rev_tail,
        rev_fusion,
        **kwargs,
    )
    rev_prefix_best = prefix_best_hits(rev_hits_ending_at)

    m = len(fusion)
    suffix_best = [None] * m

    for rev_hit in rev_prefix_best:
        if rev_hit is None:
            continue

        fusion_start = m - 1 - rev_hit.fusion_end
        fusion_end = m - 1 - rev_hit.fusion_start

        protein_start = len(tail) - 1 - rev_hit.protein_end
        protein_end = len(tail) - 1 - rev_hit.protein_start

        original_pairs = []
        for p in rev_hit.aligned_pairs or []:
            original_pairs.append({
                "protein_index_0ind": len(tail) - 1 - p["protein_index_0ind"],
                "fusion_index_0ind": m - 1 - p["fusion_index_0ind"],
                "protein_aa": p["protein_aa"],
                "fusion_aa": p["fusion_aa"],
            })

        original_pairs = sorted(
            original_pairs,
            key=lambda x: x["fusion_index_0ind"],
        )

        original_hit = LocalHit(
            score=rev_hit.score,
            fusion_start=fusion_start,
            fusion_end=fusion_end,
            protein_start=protein_start,
            protein_end=protein_end,
            aligned_pairs=original_pairs,
        )

        suffix_best[fusion_start] = original_hit

    current = None
    out = [None] * m

    for j in range(m - 1, -1, -1):
        hit = suffix_best[j]
        if hit is not None and (current is None or hit.score > current.score):
            current = hit
        out[j] = current

    return out


def find_fusion_breakpoint(
    fusion: str,
    head: str,
    tail: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
    inter_fragment_gap_penalty: int = 1,
):
    """Infer the best head/tail breakpoint in a fusion sequence."""
    head_hits_exact = smith_waterman_hits_by_fusion_end(
        head,
        fusion,
        match=match,
        mismatch=mismatch,
        gap=gap,
    )
    head_best = prefix_best_hits(head_hits_exact)

    tail_best = suffix_best_hits_tail(
        tail,
        fusion,
        match=match,
        mismatch=mismatch,
        gap=gap,
    )

    best_result: Optional[FusionBreakpointResult] = None

    for k in range(len(fusion) - 1):
        h = head_best[k]
        t = tail_best[k + 1]

        if h is None or t is None:
            continue

        if h.fusion_end >= t.fusion_start:
            continue

        gap_len = t.fusion_start - h.fusion_end - 1
        score = h.score + t.score - inter_fragment_gap_penalty * gap_len

        result = FusionBreakpointResult(
            breakpoint_after=h.fusion_end,
            score=score,
            head_hit=h,
            tail_hit=t,
        )

        if best_result is None or result.score > best_result.score:
            best_result = result

    return best_result


def get_1ind_fragment(s, start, end):
    """Get a 1-indexed inclusive fragment from a string."""
    return s[start - 1:end]


def find_fragment_bounds(seq, frag, *, use_last=False):
    """Return 1-indexed inclusive start/end coordinates of frag in seq."""
    if frag is None or frag == "":
        return None, None

    idx = seq.rfind(frag) if use_last else seq.find(frag)

    if idx == -1:
        return None, None

    start = idx + 1
    end = start + len(frag) - 1
    return start, end


def mutations_from_hit(hit):
    """Extract parent-to-fusion mismatches from an alignment hit."""
    muts = []

    for p in hit.aligned_pairs or []:
        if p["protein_aa"] != p["fusion_aa"]:
            parent_i = p["protein_index_0ind"] + 1
            fusion_i = p["fusion_index_0ind"] + 1

            muts.append({
                "parent_index_1ind": parent_i,
                "fusion_index_1ind": fusion_i,
                "parent_aa": p["protein_aa"],
                "fusion_aa": p["fusion_aa"],
                "notation": f'{p["protein_aa"]}{parent_i} -> {p["fusion_aa"]}{fusion_i}',
            })

    return muts


def mutation_list_html(muts):
    """Render mutation annotations as compact HTML."""
    if not muts:
        return "<span class='mutation-list'>None</span>"

    text = ", ".join(m["notation"] for m in muts)
    return f"<span class='mutation-list'>{escape(text)}</span>"


def color_sequence_by_positions(seq, colored_positions):
    """Color selected 0-indexed positions in a sequence."""
    pieces = []

    for i, aa in enumerate(seq):
        color = colored_positions.get(i)
        if color is None:
            pieces.append(escape(aa))
        else:
            pieces.append(
                f"<span style='color:{color}; font-weight:600'>"
                f"{escape(aa)}"
                f"</span>"
            )

    return "".join(pieces)


def color_fusion_by_hits(fusion_seq, head_hit, tail_hit, head_color, tail_color):
    """Color fusion sequence residues covered by head and tail hits."""
    colored = {}

    for p in head_hit.aligned_pairs or []:
        colored[p["fusion_index_0ind"]] = head_color

    for p in tail_hit.aligned_pairs or []:
        colored[p["fusion_index_0ind"]] = tail_color

    return color_sequence_by_positions(fusion_seq, colored)


def color_parent_by_hit(parent_seq, hit, color):
    """Color parent sequence residues covered by a hit."""
    colored = {}

    for p in hit.aligned_pairs or []:
        colored[p["protein_index_0ind"]] = color

    return color_sequence_by_positions(parent_seq, colored)


def color_bounds(seq, bounds_and_colors):
    """Color sequence regions using 1-indexed inclusive bounds."""
    pieces = []
    last0 = 0

    clean = [
        (start, end, color)
        for start, end, color in bounds_and_colors
        if start is not None and end is not None
    ]
    clean = sorted(clean, key=lambda x: x[0])

    for start, end, color in clean:
        start0 = start - 1
        end0 = end

        pieces.append(escape(seq[last0:start0]))
        pieces.append(
            f"<span style='color:{color}; font-weight:600'>"
            f"{escape(seq[start0:end0])}"
            f"</span>"
        )
        last0 = end0

    pieces.append(escape(seq[last0:]))
    return "".join(pieces)


def normalize_breakpoint_result(result, fusion_seq=None):
    """Convert supported find_fusion_breakpoint outputs into a standard dict."""
    if isinstance(result, dict):
        hfrag = (
            result.get("hfrag")
            or result.get("head_frag")
            or result.get("head_fragment")
            or result.get("predicted_head_fragment")
        )
        tfrag = (
            result.get("tfrag")
            or result.get("tail_frag")
            or result.get("tail_fragment")
            or result.get("predicted_tail_fragment")
        )

        if hfrag is None or tfrag is None:
            raise ValueError(
                f"Could not find hfrag/tfrag in result dict. Keys were: {list(result.keys())}"
            )

        out = dict(result)
        out["hfrag"] = hfrag
        out["tfrag"] = tfrag
        out["raw_result"] = result
        return out

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return {
            "hfrag": result[0],
            "tfrag": result[1],
            "raw_result": result,
        }

    if isinstance(result, pd.Series):
        return normalize_breakpoint_result(result.to_dict(), fusion_seq=fusion_seq)

    if hasattr(result, "head_hit") and hasattr(result, "tail_hit"):
        if fusion_seq is None:
            raise ValueError(
                "This result has head_hit/tail_hit coordinates, but fusion_seq was not provided."
            )

        hhit = result.head_hit
        thit = result.tail_hit

        hfrag = fusion_seq[hhit.fusion_start:hhit.fusion_end + 1]
        tfrag = fusion_seq[thit.fusion_start:thit.fusion_end + 1]

        return {
            "hfrag": hfrag,
            "tfrag": tfrag,
            "breakpoint_after": getattr(result, "breakpoint_after", None),
            "score": getattr(result, "score", None),
            "head_hit": hhit,
            "tail_hit": thit,
            "raw_result": result,
        }

    raise ValueError(
        f"Unsupported find_fusion_breakpoint() output type: {type(result)}\n"
        f"Value was:\n{result}"
    )


def validate_fragments(fusion_seq, head_seq, tail_seq, hfrag, tfrag):
    """Validate hfrag/tfrag positions in the fusion and parent proteins."""
    h_fusion_start, h_fusion_end = find_fragment_bounds(
        fusion_seq,
        hfrag,
        use_last=False,
    )
    t_fusion_start, t_fusion_end = find_fragment_bounds(
        fusion_seq,
        tfrag,
        use_last=True,
    )

    h_parent_start, h_parent_end = find_fragment_bounds(
        head_seq,
        hfrag,
        use_last=False,
    )
    t_parent_start, t_parent_end = find_fragment_bounds(
        tail_seq,
        tfrag,
        use_last=False,
    )

    checks = {
        "head_in_fusion": h_fusion_start is not None,
        "tail_in_fusion": t_fusion_start is not None,
        "head_in_head_protein": h_parent_start is not None,
        "tail_in_tail_protein": t_parent_start is not None,
        "head_before_tail": (
            h_fusion_end is not None
            and t_fusion_start is not None
            and h_fusion_end < t_fusion_start
        ),
    }
    checks["all_valid"] = all(checks.values())

    return {
        "hfrag": hfrag,
        "tfrag": tfrag,
        "fusion_head_bounds": (h_fusion_start, h_fusion_end),
        "fusion_tail_bounds": (t_fusion_start, t_fusion_end),
        "head_parent_bounds": (h_parent_start, h_parent_end),
        "tail_parent_bounds": (t_parent_start, t_parent_end),
        "checks": checks,
    }


def display_colored_fusion_result(
    fusion_seq,
    head_seq,
    tail_seq,
    validation,
    *,
    title=None,
    head_label="Head",
    tail_label="Tail",
    head_color="red",
    tail_color="blue",
):
    """Display a simple colored fusion result in a notebook, if IPython is available."""
    if display is None or IPyHTML is None:
        raise RuntimeError("IPython display is not available in this environment.")

    hfrag = validation["hfrag"]
    tfrag = validation["tfrag"]

    h_fusion_start, h_fusion_end = validation["fusion_head_bounds"]
    t_fusion_start, t_fusion_end = validation["fusion_tail_bounds"]
    h_parent_start, h_parent_end = validation["head_parent_bounds"]
    t_parent_start, t_parent_end = validation["tail_parent_bounds"]

    print(f"Start and end indices (head in fusion): {h_fusion_start} {h_fusion_end}")
    if h_fusion_start is not None:
        recalculated = get_1ind_fragment(fusion_seq, h_fusion_start, h_fusion_end)
        print("Head fragment recalculated:", recalculated)
        print("Correct:", recalculated == hfrag)
    else:
        print("Head fragment not found in fusion.")
        print("Correct:", False)

    print(f"\nStart and end indices (tail in fusion): {t_fusion_start} {t_fusion_end}")
    if t_fusion_start is not None:
        recalculated = get_1ind_fragment(fusion_seq, t_fusion_start, t_fusion_end)
        print("Tail fragment recalculated:", recalculated)
        print("Correct:", recalculated == tfrag)
    else:
        print("Tail fragment not found in fusion.")
        print("Correct:", False)

    print(f"\nStart and end indices (head in head protein): {h_parent_start} {h_parent_end}")
    print(f"Start and end indices (tail in tail protein): {t_parent_start} {t_parent_end}")
    print("All valid:", validation["checks"]["all_valid"])

    fusion_html = color_bounds(
        fusion_seq,
        [
            (h_fusion_start, h_fusion_end, head_color),
            (t_fusion_start, t_fusion_end, tail_color),
        ],
    )

    head_html = color_bounds(
        head_seq,
        [(h_parent_start, h_parent_end, head_color)],
    )

    tail_html = color_bounds(
        tail_seq,
        [(t_parent_start, t_parent_end, tail_color)],
    )

    title_html = f"<h3>{escape(title)}</h3>" if title else ""

    html = f"""
    {title_html}
    <div style="font-family: monospace; font-size: 15px; line-height: 1.7; white-space: normal;">
      <b>Fusion</b><br>
      {fusion_html}<br><br>

      <b>{escape(head_label)}</b><br>
      {head_html}<br><br>

      <b>{escape(tail_label)}</b><br>
      {tail_html}
    </div>
    """

    display(IPyHTML(html))

    fname = "fusion_result"
    if title is not None:
        fname = f"fusion_{title}_result"

    with open(f"{fname}.html", "w", encoding="utf-8") as f:
        f.write(html)


def run_fusion_breakpoint_pipeline(
    fusion_seq,
    head_seq,
    tail_seq,
    *,
    fusion_name=None,
    head_label="Head",
    tail_label="Tail",
    head_color="red",
    tail_color="blue",
    find_kwargs=None,
    display_result=True,
):
    """Run breakpoint inference, validation, and optional notebook display."""
    if find_kwargs is None:
        find_kwargs = {}

    raw_result = find_fusion_breakpoint(
        fusion_seq,
        head_seq,
        tail_seq,
        **find_kwargs,
    )

    result = normalize_breakpoint_result(raw_result, fusion_seq=fusion_seq)
    hfrag = result["hfrag"]
    tfrag = result["tfrag"]

    validation = validate_fragments(
        fusion_seq=fusion_seq,
        head_seq=head_seq,
        tail_seq=tail_seq,
        hfrag=hfrag,
        tfrag=tfrag,
    )

    result["validation"] = validation

    if display_result:
        display_colored_fusion_result(
            fusion_seq=fusion_seq,
            head_seq=head_seq,
            tail_seq=tail_seq,
            validation=validation,
            title=fusion_name,
            head_label=head_label,
            tail_label=tail_label,
            head_color=head_color,
            tail_color=tail_color,
        )

    return result


def html_anchor_id(name: str) -> str:
    """Create a safe within-page HTML anchor id."""
    name = str(name).replace("::", "_")
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return f"fusion_{name}"


def safe_filename(name: str) -> str:
    """Create a filesystem-safe filename from a fusion name."""
    name = str(name).replace("::", "_")
    name = re.sub(r"[^\w\-\.]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "fusion"


def pct(part, whole, decimals=2):
    """Calculate a percentage with NA handling."""
    if whole is None or whole == 0 or part is None:
        return pd.NA
    return round(100 * part / whole, decimals)


def add_duplicate_seq_tags(names):
    """Append _seqN tags to duplicated fusion names while leaving unique names unchanged."""
    s = pd.Series(names).astype(str)
    counts = s.value_counts()
    seen = {}

    out = []
    for name in s:
        if counts[name] == 1:
            out.append(name)
        else:
            seen[name] = seen.get(name, 0) + 1
            out.append(f"{name}_seq{seen[name]}")
    return out


BEAUTIFUL_FUSION_REPORT_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  background: #f7f7f7;
  color: #222;
}
header {
  background: white;
  padding: 32px 48px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
main {
  padding: 32px 48px;
}
h1, h2, h3 {
  margin-top: 0;
}
a {
  color: #255c99;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
.top-toc {
  display: flex;
  gap: 18px;
  list-style: none;
  padding-left: 0;
  font-weight: 700;
}
.fusion-toc {
  margin-top: 16px;
  background: #f3f6fa;
  border-radius: 14px;
  padding: 14px 18px;
}
.fusion-toc summary {
  cursor: pointer;
  font-weight: 800;
  font-size: 18px;
}
.toc {
  columns: 4;
  line-height: 1.7;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(300px, 1fr));
  gap: 20px;
}
.summary-card, .fusion-section, .sequence-card {
  background: white;
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.08);
}
.fusion-section {
  margin-top: 36px;
}
.failed-section {
  border-left: 5px solid #b00020;
}
.status {
  background: #f3f6fa;
  border-left: 4px solid #255c99;
  padding: 12px 16px;
  margin: 16px 0 24px 0;
  border-radius: 8px;
  font-size: 14px;
}
.property-box {
  background: #f8fafc;
  border-radius: 14px;
  padding: 14px 16px;
  margin: 16px 0 24px 0;
}
.property-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.property-table th {
  text-align: left;
  width: 330px;
  color: #1f3655;
  padding: 6px 8px;
  border-bottom: 1px solid #e2e8f0;
}
.property-table td {
  padding: 6px 8px;
  border-bottom: 1px solid #e2e8f0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.sequence-box, .preview-sequence {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.sequence-box {
  font-size: 15px;
  line-height: 1.7;
}
.preview-sequence {
  font-size: 11px;
  line-height: 1.35;
}
.preview-title {
  display: block;
  font-weight: 800;
  margin-bottom: 8px;
}
.summary-props {
  font-size: 12px;
  color: #475569;
  background: #f8fafc;
  border-radius: 10px;
  padding: 8px 10px;
  margin: 10px 0;
}
.mutation-box {
  font-size: 13px;
  line-height: 1.5;
  background: #f8fafc;
  border-radius: 12px;
  padding: 10px 12px;
  margin: 10px 0 20px 0;
}
.mutation-list {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}
.backtop {
  margin-top: 18px;
}
.error {
  color: #b00020;
  font-weight: 700;
}
@media (max-width: 1200px) {
  .summary-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .toc {
    columns: 2;
  }
}
@media (max-width: 800px) {
  header, main {
    padding: 24px;
  }
  .summary-grid {
    grid-template-columns: 1fr;
  }
  .toc {
    columns: 1;
  }
}
@media print {
  .fusion-section {
    break-before: page;
    page-break-before: always;
  }
  .fusion-section:first-of-type {
    break-before: auto;
    page-break-before: auto;
  }
}
"""


def analyze_one_fusion(
    fusion_seq,
    head_seq,
    tail_seq,
    fusion_name="fusion",
    *,
    head_label="Head",
    tail_label="Tail",
    head_color="crimson",
    tail_color="royalblue",
    find_kwargs=None,
    decimals=2,
):
    """Analyze one fusion and return a summary row plus an HTML section."""
    if find_kwargs is None:
        find_kwargs = {}

    row = {
        "fusion_name": fusion_name,
        "fusion_sequence": fusion_seq,
        "head_sequence": head_seq,
        "tail_sequence": tail_seq,
        "status": "ok",
        "error": pd.NA,
        "fusion_anchor_id": html_anchor_id(fusion_name),
    }

    try:
        raw_result = find_fusion_breakpoint(
            fusion_seq,
            head_seq,
            tail_seq,
            **find_kwargs,
        )

        if raw_result is None:
            raise ValueError("find_fusion_breakpoint returned None")

        result = normalize_breakpoint_result(raw_result, fusion_seq=fusion_seq)

        hhit = result["head_hit"]
        thit = result["tail_hit"]

        head_mutations = mutations_from_hit(hhit)
        tail_mutations = mutations_from_hit(thit)

        hfrag = result["hfrag"]
        tfrag = result["tfrag"]

        validation = validate_fragments(
            fusion_seq=fusion_seq,
            head_seq=head_seq,
            tail_seq=tail_seq,
            hfrag=hfrag,
            tfrag=tfrag,
        )

        h_fusion_start, h_fusion_end = validation["fusion_head_bounds"]
        t_fusion_start, t_fusion_end = validation["fusion_tail_bounds"]
        h_parent_start, h_parent_end = validation["head_parent_bounds"]
        t_parent_start, t_parent_end = validation["tail_parent_bounds"]

        hfrag_len = len(hfrag)
        tfrag_len = len(tfrag)

        fusion_html = color_fusion_by_hits(
            fusion_seq=fusion_seq,
            head_hit=hhit,
            tail_hit=thit,
            head_color=head_color,
            tail_color=tail_color,
        )

        head_html = color_parent_by_hit(
            parent_seq=head_seq,
            hit=hhit,
            color=head_color,
        )

        tail_html = color_parent_by_hit(
            parent_seq=tail_seq,
            hit=thit,
            color=tail_color,
        )

        row.update({
            "head_fragment_sequence": hfrag,
            "tail_fragment_sequence": tfrag,
            "head_fusion_start_1ind": h_fusion_start,
            "head_fusion_end_1ind": h_fusion_end,
            "tail_fusion_start_1ind": t_fusion_start,
            "tail_fusion_end_1ind": t_fusion_end,
            "head_parent_start_1ind": h_parent_start,
            "head_parent_end_1ind": h_parent_end,
            "tail_parent_start_1ind": t_parent_start,
            "tail_parent_end_1ind": t_parent_end,
            "head_fragment_length": hfrag_len,
            "tail_fragment_length": tfrag_len,
            "fusion_length": len(fusion_seq),
            "head_length": len(head_seq),
            "tail_length": len(tail_seq),
            "pct_fusion_from_head": pct(hfrag_len, len(fusion_seq), decimals),
            "pct_fusion_from_tail": pct(tfrag_len, len(fusion_seq), decimals),
            "pct_head_from_head_fragment": pct(hfrag_len, len(head_seq), decimals),
            "pct_tail_from_tail_fragment": pct(tfrag_len, len(tail_seq), decimals),
            "breakpoint_after_0ind": result.get("breakpoint_after"),
            "score": result.get("score"),
            "all_valid": validation["checks"]["all_valid"],
            "fusion_colored_html": fusion_html,
            "total_head_mutations": len(head_mutations),
            "total_tail_mutations": len(tail_mutations),
            "head_mutations_json": json.dumps(head_mutations),
            "tail_mutations_json": json.dumps(tail_mutations),
        })

        html = f"""
        <section id="{html_anchor_id(fusion_name)}" class="fusion-section">
          <h2>{escape(str(fusion_name))}</h2>

          <div class="status">
            <b>Length:</b> {len(fusion_seq)} aa |
            <b>Head in fusion:</b> {h_fusion_start}–{h_fusion_end} |
            <b>Tail in fusion:</b> {t_fusion_start}–{t_fusion_end} |
            <b>All valid:</b> {validation["checks"]["all_valid"]}
          </div>

          <div class="property-box">
            <h3>Index summary</h3>
            <table class="property-table">
              <tr><th>Start and end indices: head in fusion</th><td>{h_fusion_start}–{h_fusion_end}</td></tr>
              <tr><th>Start and end indices: tail in fusion</th><td>{t_fusion_start}–{t_fusion_end}</td></tr>
              <tr><th>Start and end indices: head in head protein</th><td>{h_parent_start}–{h_parent_end}</td></tr>
              <tr><th>Start and end indices: tail in tail protein</th><td>{t_parent_start}–{t_parent_end}</td></tr>
              <tr><th>All valid</th><td>{validation["checks"]["all_valid"]}</td></tr>
              <tr><th>Total head mutations</th><td>{len(head_mutations)}</td></tr>
              <tr><th>Total tail mutations</th><td>{len(tail_mutations)}</td></tr>
            </table>
          </div>

          <div class="mutation-box">
            <b>Head mutations:</b> {mutation_list_html(head_mutations)}<br>
            <b>Tail mutations:</b> {mutation_list_html(tail_mutations)}
          </div>

          <div class="property-box">
            <h3>Quick stats</h3>
            <table class="property-table">
              <tr><th>% of fusion sequence made up by head fragment</th><td>{row["pct_fusion_from_head"]}%</td></tr>
              <tr><th>% of fusion sequence made up by tail fragment</th><td>{row["pct_fusion_from_tail"]}%</td></tr>
              <tr><th>% of head sequence made up by head fragment</th><td>{row["pct_head_from_head_fragment"]}%</td></tr>
              <tr><th>% of tail sequence made up by tail fragment</th><td>{row["pct_tail_from_tail_fragment"]}%</td></tr>
            </table>
          </div>

          <div class="sequence-card">
            <h3>Colored sequences</h3>
            <div class="sequence-box">
              <b>Fusion</b><br>
              {fusion_html}<br><br>

              <b>{escape(head_label)}</b><br>
              {head_html}<br><br>

              <b>{escape(tail_label)}</b><br>
              {tail_html}
            </div>
          </div>

          <p class="backtop"><a href="#top">Back to top</a></p>
        </section>
        """

        return row, html

    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)

        html = f"""
        <section id="{html_anchor_id(fusion_name)}" class="fusion-section failed-section">
          <h2>{escape(str(fusion_name))}</h2>
          <p class="error"><b>Failed:</b> {escape(str(e))}</p>
          <p class="backtop"><a href="#top">Back to top</a></p>
        </section>
        """

        return row, html


def run_fusion_breakpoint_batch(
    df,
    *,
    fusion_col="fusion",
    head_col="head",
    tail_col="tail",
    fusion_name_col="fusion_name",
    out_dir="fusion_breakpoint_results",
    combined_html_name="all_fusion_results.html",
    combined_csv_name="all_fusion_results.csv",
    write_per_fusion_html=True,
    display_combined=False,
    head_color="crimson",
    tail_color="royalblue",
    find_kwargs=None,
    decimals=2,
):
    """Run breakpoint analysis over a dataframe and write combined HTML/CSV outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    individual_out_dir = out_dir / "individual_fusions"
    individual_out_dir.mkdir(parents=True, exist_ok=True)

    required = [fusion_col, head_col, tail_col, fusion_name_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    working = df.copy()
    working["_display_fusion_name"] = add_duplicate_seq_tags(working[fusion_name_col])

    rows = []
    html_sections = []

    for _, r in working.iterrows():
        fusion_name = r["_display_fusion_name"]

        row, section_html = analyze_one_fusion(
            fusion_seq=str(r[fusion_col]),
            head_seq=str(r[head_col]),
            tail_seq=str(r[tail_col]),
            fusion_name=fusion_name,
            head_color=head_color,
            tail_color=tail_color,
            find_kwargs=find_kwargs,
            decimals=decimals,
        )

        rows.append(row)
        html_sections.append(section_html)

        if write_per_fusion_html:
            per_file = individual_out_dir / f"{safe_filename(fusion_name)}.html"
            with open(per_file, "w", encoding="utf-8") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                  <meta charset="utf-8">
                  <title>{escape(str(fusion_name))}</title>
                  <style>
                    {BEAUTIFUL_FUSION_REPORT_CSS}
                  </style>
                </head>
                <body>
                  <header id="top">
                    <h1>{escape(str(fusion_name))}</h1>
                  </header>
                  <main>
                    {section_html}
                  </main>
                </body>
                </html>
                """)

    results_df = pd.DataFrame(rows)

    toc_items = sorted(
        [
            (str(row["fusion_name"]), html_anchor_id(row["fusion_name"]))
            for row in rows
        ],
        key=lambda x: x[0].lower(),
    )

    fusion_toc_html = "\n".join(
        f'<li><a href="#{anchor}">{escape(name)}</a></li>'
        for name, anchor in toc_items
    )

    preview_items = sorted(
        [
            (
                str(row["fusion_name"]),
                str(row.get("fusion_colored_html", "")),
                row.get("fusion_length", pd.NA),
                row.get("head_fusion_start_1ind", pd.NA),
                row.get("head_fusion_end_1ind", pd.NA),
                row.get("tail_fusion_start_1ind", pd.NA),
                row.get("tail_fusion_end_1ind", pd.NA),
                row.get("pct_fusion_from_head", pd.NA),
                row.get("pct_fusion_from_tail", pd.NA),
                row.get("total_head_mutations", 0),
                row.get("total_tail_mutations", 0),
            )
            for row in rows
            if row.get("status") == "ok"
        ],
        key=lambda x: x[0].lower(),
    )

    preview_html = "\n".join(
        f"""
        <div class="summary-card">
          <a class="preview-title" href="#{html_anchor_id(name)}">{escape(name)}</a>

          <div class="summary-props">
            <b>Length:</b> {fusion_length} aa<br>
            <b>Head:</b> {hstart}–{hend} ({hpct}%)<br>
            <b>Tail:</b> {tstart}–{tend} ({tpct}%)<br>
            <b>Head mutations:</b> {n_head_mutations}<br>
            <b>Tail mutations:</b> {n_tail_mutations}
          </div>

          <div class="preview-sequence">{colored_fusion}</div>
        </div>
        """
        for (
            name,
            colored_fusion,
            fusion_length,
            hstart,
            hend,
            tstart,
            tend,
            hpct,
            tpct,
            n_head_mutations,
            n_tail_mutations,
        ) in preview_items
    )

    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Fusion breakpoint results</title>
      <style>
        {BEAUTIFUL_FUSION_REPORT_CSS}
      </style>
    </head>

    <body>
      <header id="top">
        <h1>Fusion breakpoint results</h1>

        <h2>Table of contents</h2>
        <ul class="top-toc">
          <li><a href="#summary">Summary</a></li>
          <li><a href="#fusion_sections">Fusion sections</a></li>
          <li><a href="#colored_fusion_preview">Colored fusion preview</a></li>
        </ul>

        <details class="fusion-toc">
          <summary>Fusion proteins ({len(rows)})</summary>
          <ol class="toc">
            {fusion_toc_html}
          </ol>
        </details>
      </header>

      <main>
        <section id="summary" class="fusion-section">
          <h2>Summary</h2>
          <div class="status">
            <b>Total fusion proteins:</b> {len(rows)} |
            <b>Successful:</b> {sum(row.get("status") == "ok" for row in rows)} |
            <b>Failed:</b> {sum(row.get("status") != "ok" for row in rows)}
          </div>
        </section>

        <section id="colored_fusion_preview" class="fusion-section">
          <h2>Colored fusion preview</h2>
          <p>
            Compact overview of the predicted head and tail regions within each fusion sequence.
          </p>
          <div class="summary-grid">
            {preview_html}
          </div>
        </section>

        <section id="fusion_sections">
          {''.join(html_sections)}
        </section>
      </main>
    </body>
    </html>
    """

    html_path = out_dir / combined_html_name
    csv_path = out_dir / combined_csv_name

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(combined_html)

    results_df.to_csv(csv_path, index=False)

    if display_combined and display is not None and IPyHTML is not None:
        display(IPyHTML(combined_html))

    return results_df, str(html_path), str(csv_path)