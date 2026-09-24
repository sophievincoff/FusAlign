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

    # Store metadata first, then traceback once per fusion end for speed.
    best_meta = [None] * m

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
                old = best_meta[j - 1]
                if old is None or score > old["score"]:
                    best_meta[j - 1] = {
                        "score": score,
                        "i_end_1ind": i,
                        "j_end_1ind": j,
                        "start": st,
                    }

    best_ending_at = [None] * m
    for j0, meta in enumerate(best_meta):
        if meta is None:
            continue
        protein_start, fusion_start = meta["start"]
        i_end_1ind = meta["i_end_1ind"]
        j_end_1ind = meta["j_end_1ind"]
        aligned_pairs = traceback_aligned_pairs(
            dp=dp,
            tb=tb,
            protein=protein,
            fusion=fusion,
            i=i_end_1ind,
            j=j_end_1ind,
        )
        best_ending_at[j0] = LocalHit(
            score=meta["score"],
            fusion_start=fusion_start,
            fusion_end=j0,
            protein_start=protein_start,
            protein_end=i_end_1ind - 1,
            aligned_pairs=aligned_pairs,
        )

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

    return choose_breakpoint(
        head_best,
        tail_best,
        inter_fragment_gap_penalty=inter_fragment_gap_penalty,
    )


def choose_breakpoint(head_best, tail_best, inter_fragment_gap_penalty: int = 1):
    """Pick the best head/tail breakpoint from precomputed per-position hits.

    ``head_best[k]`` is the best head hit ending at or before fusion index k.
    ``tail_best[k]`` is the best tail hit starting at or after fusion index k.
    Both lists are the length of the fusion sequence.
    """
    best_result: Optional[FusionBreakpointResult] = None
    n_positions = min(len(head_best), len(tail_best))

    for k in range(n_positions - 1):
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


MUTATION_COLOR = "#7c3aed"

RESIDUE_QUERY_RE = re.compile(
    r"^\s*(?P<partner>[A-Za-z][A-Za-z0-9_.-]*)\s*[:\s]\s*"
    r"(?:(?P<aa>[A-Za-z])\s*)?(?P<pos>\d+)"
    r"(?:\s*(?:-|\.\.|to)\s*(?:(?P<aa_end>[A-Za-z])\s*)?(?P<pos_end>\d+))?\s*$",
    re.IGNORECASE,
)


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


def _hit_span_1ind(hit):
    """Return 1-indexed inclusive parent and fusion spans for a hit."""
    if hit is None:
        return None
    return {
        "parent_start_1ind": hit.protein_start + 1,
        "parent_end_1ind": hit.protein_end + 1,
        "fusion_start_1ind": hit.fusion_start + 1,
        "fusion_end_1ind": hit.fusion_end + 1,
    }


def build_residue_maps(head_hit, tail_hit):
    """Build JSON-serializable parent↔fusion residue maps from alignment hits."""
    maps = {
        "head": {},
        "tail": {},
        "fusion": {},
        "spans": {
            "head": _hit_span_1ind(head_hit),
            "tail": _hit_span_1ind(tail_hit),
        },
    }

    for partner, hit in (("head", head_hit), ("tail", tail_hit)):
        if hit is None:
            continue
        for p in hit.aligned_pairs or []:
            parent_i = p["protein_index_0ind"] + 1
            fusion_i = p["fusion_index_0ind"] + 1
            entry = {
                "partner": partner,
                "parent_index_1ind": parent_i,
                "fusion_index_1ind": fusion_i,
                "parent_aa": p["protein_aa"],
                "fusion_aa": p["fusion_aa"],
                "is_mutation": p["protein_aa"] != p["fusion_aa"],
            }
            maps[partner][str(parent_i)] = entry
            # Prefer first write; overlapping head/tail on fusion is rare for valid BP.
            maps["fusion"].setdefault(str(fusion_i), entry)

    return maps


def _normalize_partner(partner, partner_aliases=None):
    """Normalize partner token to head/tail/fusion, with optional alias support."""
    norm = str(partner).strip().lower()
    aliases = {"head": "head", "tail": "tail", "fusion": "fusion"}
    if partner_aliases:
        aliases.update({str(k).strip().lower(): v for k, v in partner_aliases.items()})
    canonical = aliases.get(norm)
    if canonical not in {"head", "tail", "fusion"}:
        allowed = sorted(aliases.keys())
        raise ValueError(
            "Partner must be Head, Tail, or Fusion"
            + (f" (or alias: {', '.join(allowed)})" if partner_aliases else "")
            + f". Got {partner!r}."
        )
    return canonical


def parse_residue_query(text, *, partner_aliases=None):
    """
    Parse queries like ``Tail:Y1078``, ``Tail Y1078``, ``head:45``, ``Fusion:Y450``,
    and ranges like ``Tail:1078-1085``.

    Returns dict with partner, position (1-based), and optional expected_aa.
    """
    if text is None or not str(text).strip():
        raise ValueError("Empty residue query.")

    match = RESIDUE_QUERY_RE.match(str(text))
    if not match:
        raise ValueError(
            "Could not parse query. Examples: Tail:Y1078, Head:S12, Fusion:Y200, Tail:1078"
        )

    partner = _normalize_partner(match.group("partner"), partner_aliases=partner_aliases)
    position = int(match.group("pos"))
    position_end = int(match.group("pos_end")) if match.group("pos_end") else position
    aa = match.group("aa")
    expected_aa = aa.upper() if aa else None
    aa_end = match.group("aa_end")
    expected_aa_end = aa_end.upper() if aa_end else None
    if expected_aa is not None and not expected_aa.isalpha():
        raise ValueError(f"Invalid amino-acid letter: {aa!r}")
    if expected_aa_end is not None and not expected_aa_end.isalpha():
        raise ValueError(f"Invalid amino-acid letter: {aa_end!r}")
    if position_end < position:
        raise ValueError("Range end must be >= range start.")
    if position_end != position and (expected_aa is not None or expected_aa_end is not None):
        raise ValueError("For ranges, omit amino-acid letters (example: Tail:1078-1085).")

    return {
        "partner": partner,
        "position": position,
        "position_end": position_end,
        "expected_aa": expected_aa,
    }


def format_residue_query_answer(answer):
    """Render a query_residue() result as Markdown."""
    status = answer.get("status")

    if status == "range":
        start = answer.get("start")
        end = answer.get("end")
        partner = answer.get("partner")
        lines = [f"**Range query `{partner}:{start}-{end}`**", ""]
        counts = answer.get("counts", {})
        lines.append(
            "- Summary: "
            f"{counts.get('found', 0)} found, "
            f"{counts.get('not_in_alignment', 0)} not in alignment, "
            f"{counts.get('parent_aa_mismatch', 0)} amino-acid disagreements, "
            f"{counts.get('error', 0)} errors."
        )
        lines.append("")
        for item in answer.get("results", []):
            label = item.get("query")
            item_status = item.get("status")
            if item_status == "found":
                lines.append(
                    f"- `{label}` -> {item['fusion_aa']}{item['fusion_index_1ind']} "
                    f"(from {item['parent_aa']}{item['parent_index_1ind']})"
                )
            else:
                lines.append(
                    f"- `{label}` -> {item_status.replace('_', ' ')}: {item.get('message', '')}"
                )
        return "\n".join(lines)
    if status == "found":
        lines = [
            f"**{answer['summary']}**",
            "",
            f"- Partner: `{answer['partner']}`",
            f"- Parent: `{answer['parent_aa']}{answer['parent_index_1ind']}`",
            f"- Fusion: `{answer['fusion_aa']}{answer['fusion_index_1ind']}`",
            f"- Amino acid match: **{'yes' if answer['aa_matches'] else 'no'}**",
        ]
        if answer.get("is_mutation"):
            lines.append("- Note: this aligned pair is a **substitution** in the fusion.")
        if answer.get("warning"):
            lines.append(f"- Warning: {answer['warning']}")
        return "\n".join(lines)

    if status == "not_in_alignment":
        span = answer.get("hit_span") or {}
        span_txt = ""
        if span:
            span_txt = (
                f" Aligned `{answer.get('partner')}` span on parent is "
                f"{span.get('parent_start_1ind')}–{span.get('parent_end_1ind')}; "
                f"on fusion {span.get('fusion_start_1ind')}–{span.get('fusion_end_1ind')}."
            )
        return f"**Not in alignment.** {answer.get('message', '')}{span_txt}"

    if status == "parent_aa_mismatch":
        return (
            f"**Parent amino acid disagreement.** {answer.get('message', '')}\n\n"
            f"- Requested: `{answer.get('expected_aa')}{answer.get('parent_index_1ind')}`\n"
            f"- Observed at that parent position: "
            f"`{answer.get('parent_aa')}{answer.get('parent_index_1ind')}`\n"
            f"- Mapped fusion site: "
            f"`{answer.get('fusion_aa')}{answer.get('fusion_index_1ind')}`"
        )

    if status == "error":
        return f"**Query error.** {answer.get('message', '')}"

    return f"**Unexpected result:** `{answer}`"


def _query_residue_single(maps, *, partner, position, expected_aa=None):
    """Single-position residue query implementation."""
    partner = str(partner).lower()
    position = int(position)
    if expected_aa is not None:
        expected_aa = str(expected_aa).upper()

    if partner not in {"head", "tail", "fusion"}:
        return {
            "status": "error",
            "message": f"Partner must be Head, Tail, or Fusion (got {partner!r}).",
        }

    if partner in {"head", "tail"}:
        entry = (maps.get(partner) or {}).get(str(position))
        span = (maps.get("spans") or {}).get(partner)
        if entry is None:
            return {
                "status": "not_in_alignment",
                "partner": partner,
                "parent_index_1ind": position,
                "expected_aa": expected_aa,
                "hit_span": span,
                "message": (
                    f"{partner.capitalize()} residue {position} is not covered by "
                    f"the {partner} alignment hit."
                ),
            }
        if expected_aa is not None and entry["parent_aa"] != expected_aa:
            return {
                "status": "parent_aa_mismatch",
                "partner": partner,
                "parent_index_1ind": entry["parent_index_1ind"],
                "fusion_index_1ind": entry["fusion_index_1ind"],
                "parent_aa": entry["parent_aa"],
                "fusion_aa": entry["fusion_aa"],
                "expected_aa": expected_aa,
                "is_mutation": entry["is_mutation"],
                "message": (
                    f"Requested {expected_aa}{position}, but the {partner} sequence "
                    f"has {entry['parent_aa']} at that position."
                ),
            }
        aa_matches = expected_aa is None or entry["fusion_aa"] == expected_aa
        warning = None
        if expected_aa is not None and entry["fusion_aa"] != expected_aa:
            warning = (
                f"Fusion has {entry['fusion_aa']}{entry['fusion_index_1ind']} "
                f"(not {expected_aa})."
            )
        summary = (
            f"{partner.capitalize()} {entry['parent_aa']}{entry['parent_index_1ind']} "
            f"-> fusion {entry['fusion_aa']}{entry['fusion_index_1ind']}"
        )
        return {
            "status": "found",
            "partner": partner,
            "parent_index_1ind": entry["parent_index_1ind"],
            "fusion_index_1ind": entry["fusion_index_1ind"],
            "parent_aa": entry["parent_aa"],
            "fusion_aa": entry["fusion_aa"],
            "expected_aa": expected_aa,
            "aa_matches": aa_matches,
            "is_mutation": entry["is_mutation"],
            "summary": summary,
            "warning": warning,
        }

    entry = (maps.get("fusion") or {}).get(str(position))
    if entry is None:
        return {
            "status": "not_in_alignment",
            "partner": "fusion",
            "fusion_index_1ind": position,
            "expected_aa": expected_aa,
            "hit_span": None,
            "message": (
                f"Fusion residue {position} is not covered by the head or tail "
                f"alignment hits."
            ),
        }
    if expected_aa is not None and entry["fusion_aa"] != expected_aa:
        return {
            "status": "parent_aa_mismatch",
            "partner": entry["partner"],
            "parent_index_1ind": entry["parent_index_1ind"],
            "fusion_index_1ind": entry["fusion_index_1ind"],
            "parent_aa": entry["parent_aa"],
            "fusion_aa": entry["fusion_aa"],
            "expected_aa": expected_aa,
            "is_mutation": entry["is_mutation"],
            "message": (
                f"Requested fusion {expected_aa}{position}, but the fusion sequence "
                f"has {entry['fusion_aa']} at that position."
            ),
        }
    aa_matches = expected_aa is None or entry["fusion_aa"] == expected_aa
    summary = (
        f"Fusion {entry['fusion_aa']}{entry['fusion_index_1ind']} -> "
        f"{entry['partner']} {entry['parent_aa']}{entry['parent_index_1ind']}"
    )
    return {
        "status": "found",
        "partner": entry["partner"],
        "parent_index_1ind": entry["parent_index_1ind"],
        "fusion_index_1ind": entry["fusion_index_1ind"],
        "parent_aa": entry["parent_aa"],
        "fusion_aa": entry["fusion_aa"],
        "expected_aa": expected_aa,
        "aa_matches": aa_matches,
        "is_mutation": entry["is_mutation"],
        "summary": summary,
        "warning": None,
    }


def query_residue(
    maps,
    *,
    partner=None,
    position=None,
    position_end=None,
    expected_aa=None,
    query_text=None,
    partner_aliases=None,
):
    """
    Map a residue between parent and fusion using maps from build_residue_maps().

    Provide either ``query_text`` (parsed) or explicit partner/position/expected_aa.
    """
    try:
        if query_text is not None:
            parsed = parse_residue_query(query_text, partner_aliases=partner_aliases)
            partner = parsed["partner"]
            position = parsed["position"]
            position_end = parsed.get("position_end", position)
            expected_aa = parsed["expected_aa"]
        if partner is None or position is None:
            raise ValueError("partner and position are required.")
        partner = _normalize_partner(partner, partner_aliases=partner_aliases)
        position = int(position)
        position_end = int(position if position_end is None else position_end)
        if position_end < position:
            raise ValueError("position_end must be >= position.")
        if position_end != position and expected_aa is not None:
            raise ValueError("expected_aa is only supported for single-position queries.")
        if expected_aa is not None:
            expected_aa = str(expected_aa).upper()
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if position_end == position:
        return _query_residue_single(
            maps,
            partner=partner,
            position=position,
            expected_aa=expected_aa,
        )

    if position_end - position + 1 > 2000:
        return {
            "status": "error",
            "message": "Range is too large (>2000 residues).",
        }

    items = []
    counts = {
        "found": 0,
        "not_in_alignment": 0,
        "parent_aa_mismatch": 0,
        "error": 0,
    }
    for pos in range(position, position_end + 1):
        ans = _query_residue_single(maps, partner=partner, position=pos, expected_aa=None)
        ans["query"] = f"{partner}:{pos}"
        items.append(ans)
        if ans["status"] in counts:
            counts[ans["status"]] += 1
        else:
            counts["error"] += 1

    return {
        "status": "range",
        "partner": partner,
        "start": position,
        "end": position_end,
        "results": items,
        "counts": counts,
    }


def mutation_list_html(muts):
    """Render mutation annotations as compact HTML."""
    if not muts:
        return "<span class='mutation-list'>None</span>"

    text = ", ".join(m["notation"] for m in muts)
    return f"<span class='mutation-list'>{escape(text)}</span>"


def color_sequence_by_positions(
    seq,
    colored_positions,
    mutation_positions=None,
    mutation_color=MUTATION_COLOR,
):
    """
    Color selected 0-indexed positions in a sequence.

    ``colored_positions`` maps index → CSS color for regional (head/tail) paint.
    ``mutation_positions`` is a set of 0-indexed indices styled purple + underline
    and takes priority over regional color.
    """
    mutation_positions = mutation_positions or set()
    pieces = []

    for i, aa in enumerate(seq):
        if i in mutation_positions:
            pieces.append(
                f"<span class='mut' style='color:{mutation_color}; "
                f"font-weight:600; text-decoration:underline'>"
                f"{escape(aa)}"
                f"</span>"
            )
        else:
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


def color_fusion_by_hits(
    fusion_seq,
    head_hit,
    tail_hit,
    head_color,
    tail_color,
    mutation_color=MUTATION_COLOR,
):
    """Color fusion sequence residues; substitutions are purple + underlined."""
    colored = {}
    mutations = set()

    for p in head_hit.aligned_pairs or []:
        idx = p["fusion_index_0ind"]
        colored[idx] = head_color
        if p["protein_aa"] != p["fusion_aa"]:
            mutations.add(idx)

    for p in tail_hit.aligned_pairs or []:
        idx = p["fusion_index_0ind"]
        colored[idx] = tail_color
        if p["protein_aa"] != p["fusion_aa"]:
            mutations.add(idx)

    return color_sequence_by_positions(
        fusion_seq,
        colored,
        mutations,
        mutation_color=mutation_color,
    )


def color_parent_by_hit(parent_seq, hit, color, mutation_color=MUTATION_COLOR):
    """Color parent residues covered by a hit; substitutions purple + underlined."""
    colored = {}
    mutations = set()

    for p in hit.aligned_pairs or []:
        idx = p["protein_index_0ind"]
        colored[idx] = color
        if p["protein_aa"] != p["fusion_aa"]:
            mutations.add(idx)

    return color_sequence_by_positions(
        parent_seq,
        colored,
        mutations,
        mutation_color=mutation_color,
    )


def sequence_color_legend_html(head_color, tail_color, mutation_color=MUTATION_COLOR):
    """HTML legend for head / tail / mutation sequence coloring."""
    return f"""
    <div class="seq-legend">
      <span class="legend-item">
        <span class="legend-swatch" style="background:{escape(head_color)}"></span> Head
      </span>
      <span class="legend-item">
        <span class="legend-swatch" style="background:{escape(tail_color)}"></span> Tail
      </span>
      <span class="legend-item">
        <span class="legend-swatch" style="background:{escape(mutation_color)}"></span>
        <span style="color:{escape(mutation_color)}; text-decoration:underline;">Mutation</span>
      </span>
    </div>
    """


def residue_maps_by_fusion_name(rows):
    """Build fusion_name → residue_maps dict from analyze_one_fusion row dicts."""
    payload = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        raw = row.get("residue_maps_json")
        if not raw:
            continue
        try:
            payload[str(row["fusion_name"])] = (
                json.loads(raw) if isinstance(raw, str) else raw
            )
        except (TypeError, json.JSONDecodeError):
            continue
    return payload


RESIDUE_QUERY_WIDGET_JS = r"""
(function () {
  function parseQuery(text) {
    var m = /^\s*(head|tail|fusion)\s*[:\s]\s*(?:([A-Za-z])\s*)?(\d+)\s*$/i.exec(text || "");
    if (!m) throw new Error("Could not parse query. Examples: Tail:Y1078, Head:S12, Fusion:Y200");
    return {
      partner: m[1].toLowerCase(),
      expected_aa: m[2] ? m[2].toUpperCase() : null,
      position: parseInt(m[3], 10)
    };
  }

  function spanText(partner, span) {
    if (!span) return "";
    return " Aligned " + partner + " span on parent is " +
      span.parent_start_1ind + "–" + span.parent_end_1ind +
      "; on fusion " + span.fusion_start_1ind + "–" + span.fusion_end_1ind + ".";
  }

  function queryResidue(maps, partner, position, expectedAa) {
    if (partner === "head" || partner === "tail") {
      var entry = (maps[partner] || {})[String(position)];
      var span = ((maps.spans || {})[partner]) || null;
      if (!entry) {
        return "Not in alignment. " + partner.charAt(0).toUpperCase() + partner.slice(1) +
          " residue " + position + " is not covered by the " + partner + " alignment hit." +
          spanText(partner, span);
      }
      if (expectedAa && entry.parent_aa !== expectedAa) {
        return "Parent amino acid disagreement. Requested " + expectedAa + position +
          ", but the " + partner + " sequence has " + entry.parent_aa + " at that position.\n" +
          "Mapped fusion site: " + entry.fusion_aa + entry.fusion_index_1ind;
      }
      var aaMatches = !expectedAa || entry.fusion_aa === expectedAa;
      var summary = partner.charAt(0).toUpperCase() + partner.slice(1) + " " +
        entry.parent_aa + entry.parent_index_1ind + " → fusion " +
        entry.fusion_aa + entry.fusion_index_1ind;
      var lines = [summary,
        "Parent: " + entry.parent_aa + entry.parent_index_1ind,
        "Fusion: " + entry.fusion_aa + entry.fusion_index_1ind,
        "Amino acid match: " + (aaMatches ? "yes" : "no")];
      if (entry.is_mutation) lines.push("Note: this aligned pair is a substitution in the fusion.");
      if (expectedAa && entry.fusion_aa !== expectedAa) {
        lines.push("Warning: Fusion has " + entry.fusion_aa + entry.fusion_index_1ind +
          " (not " + expectedAa + ").");
      }
      return lines.join("\n");
    }

    var fEntry = (maps.fusion || {})[String(position)];
    if (!fEntry) {
      return "Not in alignment. Fusion residue " + position +
        " is not covered by the head or tail alignment hits.";
    }
    if (expectedAa && fEntry.fusion_aa !== expectedAa) {
      return "Parent amino acid disagreement. Requested fusion " + expectedAa + position +
        ", but the fusion sequence has " + fEntry.fusion_aa + " at that position.\n" +
        "Mapped parent site: " + fEntry.partner + " " + fEntry.parent_aa + fEntry.parent_index_1ind;
    }
    return "Fusion " + fEntry.fusion_aa + fEntry.fusion_index_1ind + " → " +
      fEntry.partner + " " + fEntry.parent_aa + fEntry.parent_index_1ind + "\n" +
      "Amino acid match: " + ((!expectedAa || fEntry.fusion_aa === expectedAa) ? "yes" : "no");
  }

  function runQuery() {
    var select = document.getElementById("rq-fusion");
    var input = document.getElementById("rq-input");
    var out = document.getElementById("rq-answer");
    if (!select || !input || !out) return;
    var name = select.value;
    var maps = (window.FUSALIGN_RESIDUE_MAPS || {})[name];
    if (!maps) {
      out.textContent = "No residue maps for the selected fusion.";
      return;
    }
    try {
      var parsed = parseQuery(input.value);
      out.textContent = queryResidue(maps, parsed.partner, parsed.position, parsed.expected_aa);
    } catch (err) {
      out.textContent = "Query error. " + (err && err.message ? err.message : String(err));
    }
  }

  function init() {
    var btn = document.getElementById("rq-run");
    var input = document.getElementById("rq-input");
    if (btn) btn.addEventListener("click", runQuery);
    if (input) input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") runQuery();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""


def residue_query_widget_html(maps_by_name):
    """Offline residue-query panel + embedded maps JSON for the HTML report."""
    names = sorted(maps_by_name.keys(), key=lambda s: s.lower())
    options = "\n".join(
        f'<option value="{escape(name)}">{escape(name)}</option>' for name in names
    )
    maps_json = json.dumps(maps_by_name, separators=(",", ":")).replace("<", "\\u003c")
    disabled = "disabled" if not names else ""
    return f"""
    <section id="residue_query" class="fusion-section">
      <div class="residue-query-panel">
        <h3>Residue query</h3>
        <p>
          Map a parent or fusion residue without re-aligning.
          Examples: <code>Tail:Y1078</code>, <code>Head:S12</code>, <code>Fusion:Y200</code>.
        </p>
        <div class="residue-query-row">
          <label for="rq-fusion">Fusion</label>
          <select id="rq-fusion" {disabled}>
            {options if options else '<option value="">(no successful fusions)</option>'}
          </select>
          <input id="rq-input" type="text" placeholder="Tail:Y1078" {disabled}
                 aria-label="Residue query" />
          <button id="rq-run" type="button" {disabled}>Query</button>
        </div>
        <div id="rq-answer" class="residue-query-answer"></div>
      </div>
    </section>
    <script type="application/json" id="fusalign-residue-maps">{maps_json}</script>
    <script>
      window.FUSALIGN_RESIDUE_MAPS = JSON.parse(
        document.getElementById("fusalign-residue-maps").textContent
      );
    </script>
    <script>
    {RESIDUE_QUERY_WIDGET_JS}
    </script>
    """


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
.seq-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin: 8px 0 14px 0;
  font-size: 13px;
  color: #334155;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  display: inline-block;
}
.legend-mut {
  background: #7c3aed;
  box-shadow: inset 0 -2px 0 #4c1d95;
}
span.mut {
  color: #7c3aed;
  font-weight: 600;
  text-decoration: underline;
}
.residue-query-panel {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 14px 16px;
  margin: 18px 0 8px 0;
}
.residue-query-panel h3 {
  margin-bottom: 8px;
}
.residue-query-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-top: 10px;
}
.residue-query-panel select,
.residue-query-panel input[type="text"] {
  font-size: 14px;
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
}
.residue-query-panel input[type="text"] {
  min-width: 220px;
}
.residue-query-panel button {
  background: #255c99;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 7px 14px;
  font-weight: 700;
  cursor: pointer;
}
.residue-query-answer {
  margin-top: 12px;
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
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
    mutation_color=MUTATION_COLOR,
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
        residue_maps = build_residue_maps(hhit, thit)

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
            mutation_color=mutation_color,
        )

        head_html = color_parent_by_hit(
            parent_seq=head_seq,
            hit=hhit,
            color=head_color,
            mutation_color=mutation_color,
        )

        tail_html = color_parent_by_hit(
            parent_seq=tail_seq,
            hit=thit,
            color=tail_color,
            mutation_color=mutation_color,
        )

        legend_html = sequence_color_legend_html(
            head_color,
            tail_color,
            mutation_color=mutation_color,
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
            "residue_maps_json": json.dumps(residue_maps),
        })

        html = f"""
        <section id="{html_anchor_id(fusion_name)}" class="fusion-section"
                 data-fusion-name="{escape(str(fusion_name))}">
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
            {legend_html}
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
    mutation_color=MUTATION_COLOR,
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
            mutation_color=mutation_color,
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

    preview_legend = sequence_color_legend_html(
        head_color,
        tail_color,
        mutation_color=mutation_color,
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
            Substitutions are purple and underlined.
          </p>
          {preview_legend}
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