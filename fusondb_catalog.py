"""Normalized FusOn-DB × UniProt catalog.

Pairings are stored as integer links between one fusion oncoprotein and one
head UniProt entry and one tail UniProt entry. Sequences live in one table, so
a search does not load the exploded head×tail product into memory.

Alignment rows are written later by ``align_catalog`` and looked up by pairing id.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import zlib
from pathlib import Path

import pandas as pd

from fusion_breakpoint_pipeline import (
    MUTATION_COLOR,
    build_residue_maps,
    choose_breakpoint,
    color_sequence_by_positions,
    mutations_from_hit,
    normalize_breakpoint_result,
    pct,
    prefix_best_hits,
    sequence_color_legend_html,
    smith_waterman_hits_by_fusion_end,
    suffix_best_hits_tail,
    validate_fragments,
)

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parent / "examples" / "fusondb" / "fusondb_catalog.sqlite"
)
DEFAULT_ID_MAP_CSV = (
    Path(__file__).resolve().parent
    / "examples"
    / "fusondb"
    / "uniprot_query"
    / "id_map_combined.csv"
)
FUSONDB_HF = "hf://datasets/ChatterjeeLab/FusOn-DB/FusOn-DB.csv"
ID_MAP_ERROR_QUERY = "Error encountered when streaming data. Please try again later."

_SCHEMA = """
CREATE TABLE sequences (
    sequence_id INTEGER PRIMARY KEY,
    length INTEGER NOT NULL,
    sequence TEXT NOT NULL
);

CREATE TABLE uniprot (
    uniprot_row_id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    uniprot_id TEXT,
    reviewed TEXT,
    entry_name TEXT,
    protein_names TEXT,
    gene_names TEXT,
    organism TEXT,
    length REAL,
    isoform INTEGER,
    uniprot_id_full TEXT,
    sequence_id INTEGER NOT NULL
);

CREATE TABLE fusions (
    fusion_row_id INTEGER PRIMARY KEY,
    seq_id TEXT,
    fusiongenes TEXT NOT NULL,
    head_gene TEXT NOT NULL,
    tail_gene TEXT NOT NULL,
    cancers TEXT,
    primary_sources TEXT,
    secondary_sources TEXT,
    benchmark TEXT,
    length INTEGER,
    sequence_id INTEGER NOT NULL
);

CREATE TABLE pairings (
    pairing_id INTEGER PRIMARY KEY,
    fusion_row_id INTEGER NOT NULL,
    head_uniprot_row_id INTEGER NOT NULL,
    tail_uniprot_row_id INTEGER NOT NULL,
    both_reviewed INTEGER NOT NULL,
    UNIQUE (fusion_row_id, head_uniprot_row_id, tail_uniprot_row_id)
);

CREATE TABLE alignments (
    pairing_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    error TEXT,
    score INTEGER,
    breakpoint_after_0ind INTEGER,
    head_fusion_start_1ind INTEGER,
    head_fusion_end_1ind INTEGER,
    tail_fusion_start_1ind INTEGER,
    tail_fusion_end_1ind INTEGER,
    head_parent_start_1ind INTEGER,
    head_parent_end_1ind INTEGER,
    tail_parent_start_1ind INTEGER,
    tail_parent_end_1ind INTEGER,
    head_fragment_length INTEGER,
    tail_fragment_length INTEGER,
    fusion_length INTEGER,
    head_length INTEGER,
    tail_length INTEGER,
    pct_fusion_from_head REAL,
    pct_fusion_from_tail REAL,
    pct_head_from_head_fragment REAL,
    pct_tail_from_tail_fragment REAL,
    total_head_mutations INTEGER,
    total_tail_mutations INTEGER,
    all_valid INTEGER,
    residue_maps BLOB
);

CREATE TABLE catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def default_catalog_path() -> Path:
    return DEFAULT_CATALOG_PATH


def open_catalog(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"FusOn-DB catalog not found at {path}. Build it with build_catalog()."
        )
    conn = sqlite3.connect(path, timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 120000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def normalize_fusondb_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per fusion oncoprotein with head and tail gene symbols."""
    out = df.copy()
    if "head" not in out.columns or "tail" not in out.columns:
        genes = out["fusiongenes"].astype(str)
        if genes.str.contains(",", regex=False).any():
            out = out.assign(fusiongenes=genes.str.split(",")).explode("fusiongenes")
            out = out.reset_index(drop=True)
        parts = out["fusiongenes"].astype(str).str.split("::", expand=True)
        if parts.shape[1] != 2:
            raise ValueError("Expected fusion genes of the form HEAD::TAIL.")
        out["head"] = parts[0]
        out["tail"] = parts[1]
    out = out.dropna(subset=["head", "tail", "aa_seq"]).copy()
    out["head"] = out["head"].astype(str).str.strip()
    out["tail"] = out["tail"].astype(str).str.strip()
    out["fusiongenes"] = out["fusiongenes"].astype(str).str.strip()
    out["aa_seq"] = out["aa_seq"].astype(str).str.replace(r"\s+", "", regex=True)
    out = out.loc[
        out["head"].ne("") & out["tail"].ne("") & out["aa_seq"].ne("")
    ]
    return out.reset_index(drop=True)


def normalize_id_map(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out["Query"].astype(str) != ID_MAP_ERROR_QUERY]
    out = out.dropna(subset=["Query", "Sequence"]).copy()
    out["Query"] = out["Query"].astype(str).str.strip()
    out["Sequence"] = out["Sequence"].astype(str).str.replace(r"\s+", "", regex=True)
    out = out.loc[out["Query"].ne("") & out["Sequence"].ne("")]
    return out.reset_index(drop=True)


def _sql_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def _sql_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        value = value.item()
    return value


def _insert_sequences(conn, sequences) -> dict[str, int]:
    """Insert each distinct sequence once. Returns sequence text → id."""
    mapping: dict[str, int] = {}
    batch = []
    next_id = 1
    for sequence in sequences:
        if sequence in mapping:
            continue
        mapping[sequence] = next_id
        batch.append((next_id, len(sequence), sequence))
        next_id += 1
        if len(batch) >= 500:
            conn.executemany(
                "INSERT INTO sequences (sequence_id, length, sequence) VALUES (?, ?, ?)",
                batch,
            )
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT INTO sequences (sequence_id, length, sequence) VALUES (?, ?, ?)",
            batch,
        )
    return mapping


def build_catalog(
    db_path=None,
    *,
    fusondb_df: pd.DataFrame | None = None,
    id_map_df: pd.DataFrame | None = None,
    id_map_csv=None,
    overwrite: bool = False,
) -> Path:
    """Build the SQLite catalog. Sequences are stored once; pairings are links."""
    path = Path(db_path) if db_path is not None else DEFAULT_CATALOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            return path
        path.unlink()

    if fusondb_df is None:
        fusondb_df = pd.read_csv(FUSONDB_HF)
    if id_map_df is None:
        id_map_csv = Path(id_map_csv) if id_map_csv is not None else DEFAULT_ID_MAP_CSV
        id_map_df = pd.read_csv(id_map_csv)

    fusions = normalize_fusondb_frame(fusondb_df)
    id_map = normalize_id_map(id_map_df)

    started = time.time()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = FILE")
        conn.execute("PRAGMA cache_size = -400000")
        conn.executescript(_SCHEMA)

        print(
            f"Interning sequences from {len(fusions):,} fusions and {len(id_map):,} UniProt rows..."
        )
        sequence_ids = _insert_sequences(
            conn,
            list(fusions["aa_seq"]) + list(id_map["Sequence"]),
        )

        uniprot_rows = []
        for uniprot_row_id, (_, row) in enumerate(id_map.iterrows(), start=1):
            uniprot_rows.append(
                (
                    uniprot_row_id,
                    row["Query"],
                    _sql_text(row["UniProtID"]),
                    _sql_text(row["Reviewed"]),
                    _sql_text(row["Entry Name"]),
                    _sql_text(row["Protein names"]),
                    _sql_text(row["Gene Names"]),
                    _sql_text(row["Organism"]),
                    _sql_number(row["Length"]),
                    _sql_number(row["Isoform"]),
                    _sql_text(row["UniProtID_Full"]),
                    sequence_ids[row["Sequence"]],
                )
            )
        conn.executemany(
            """
            INSERT INTO uniprot (
                uniprot_row_id, query, uniprot_id, reviewed, entry_name,
                protein_names, gene_names, organism, length, isoform,
                uniprot_id_full, sequence_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            uniprot_rows,
        )

        fusion_rows = []
        for fusion_row_id, (_, row) in enumerate(fusions.iterrows(), start=1):
            fusion_rows.append(
                (
                    fusion_row_id,
                    _sql_text(row.get("seq_id")),
                    row["fusiongenes"],
                    row["head"],
                    row["tail"],
                    _sql_text(row.get("cancers")),
                    _sql_text(row.get("primary_sources")),
                    _sql_text(row.get("secondary_sources")),
                    _sql_text(row.get("benchmark")),
                    _sql_number(row.get("length")),
                    sequence_ids[row["aa_seq"]],
                )
            )
        conn.executemany(
            """
            INSERT INTO fusions (
                fusion_row_id, seq_id, fusiongenes, head_gene, tail_gene,
                cancers, primary_sources, secondary_sources, benchmark,
                length, sequence_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fusion_rows,
        )
        conn.commit()

        print("Linking every head UniProt hit with every tail UniProt hit...")
        conn.execute("CREATE INDEX idx_uniprot_query ON uniprot(query)")
        conn.execute("CREATE INDEX idx_fusions_head ON fusions(head_gene)")
        conn.execute("CREATE INDEX idx_fusions_tail ON fusions(tail_gene)")
        conn.execute(
            """
            INSERT INTO pairings (
                fusion_row_id, head_uniprot_row_id, tail_uniprot_row_id, both_reviewed
            )
            SELECT
                f.fusion_row_id,
                h.uniprot_row_id,
                t.uniprot_row_id,
                CASE
                    WHEN h.reviewed = 'reviewed' AND t.reviewed = 'reviewed' THEN 1
                    ELSE 0
                END
            FROM fusions f
            JOIN uniprot h ON h.query = f.head_gene
            JOIN uniprot t ON t.query = f.tail_gene
            """
        )
        conn.execute("CREATE INDEX idx_fusions_genes ON fusions(fusiongenes)")
        conn.execute("CREATE INDEX idx_fusions_seq_id ON fusions(seq_id)")
        conn.execute(
            """
            CREATE INDEX idx_pairings_reviewed
            ON pairings(fusion_row_id)
            WHERE both_reviewed = 1
            """
        )
        conn.execute("CREATE INDEX idx_uniprot_id ON uniprot(uniprot_id)")

        n_pairings = conn.execute("SELECT COUNT(*) FROM pairings").fetchone()[0]
        n_reviewed = conn.execute(
            "SELECT COUNT(*) FROM pairings WHERE both_reviewed = 1"
        ).fetchone()[0]
        n_fusions_paired = conn.execute(
            "SELECT COUNT(DISTINCT fusion_row_id) FROM pairings"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO catalog_meta (key, value) VALUES (?, ?)",
            ("built_unix", str(int(time.time()))),
        )
        conn.execute(
            "INSERT INTO catalog_meta (key, value) VALUES (?, ?)",
            ("n_pairings", str(n_pairings)),
        )
        conn.execute(
            "INSERT INTO catalog_meta (key, value) VALUES (?, ?)",
            ("n_reviewed_pairings", str(n_reviewed)),
        )
        conn.commit()
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA synchronous = NORMAL")
    finally:
        conn.close()

    elapsed = time.time() - started
    print(
        f"Wrote {path} in {elapsed:.1f}s: "
        f"{n_fusions_paired:,} oncoproteins, {n_pairings:,} pairings "
        f"({n_reviewed:,} reviewed head and tail)."
    )
    return path


def catalog_stats(conn: sqlite3.Connection) -> dict:
    def _count(sql):
        return int(conn.execute(sql).fetchone()[0])

    return {
        "sequences": _count("SELECT COUNT(*) FROM sequences"),
        "uniprot": _count("SELECT COUNT(*) FROM uniprot"),
        "fusions": _count("SELECT COUNT(*) FROM fusions"),
        "pairings": _count("SELECT COUNT(*) FROM pairings"),
        "reviewed_pairings": _count(
            "SELECT COUNT(*) FROM pairings WHERE both_reviewed = 1"
        ),
        "aligned_pairings": _count("SELECT COUNT(*) FROM alignments"),
    }


_SEARCH_SELECT = """
SELECT
    p.pairing_id,
    f.seq_id,
    f.fusiongenes,
    f.head_gene,
    f.tail_gene,
    f.cancers,
    h.uniprot_id AS head_uniprot_id,
    h.reviewed AS head_reviewed,
    h.isoform AS head_isoform,
    h.uniprot_id_full AS head_uniprot_id_full,
    h.entry_name AS head_entry_name,
    t.uniprot_id AS tail_uniprot_id,
    t.reviewed AS tail_reviewed,
    t.isoform AS tail_isoform,
    t.uniprot_id_full AS tail_uniprot_id_full,
    t.entry_name AS tail_entry_name,
    p.both_reviewed,
    a.status AS alignment_status,
    a.score,
    a.head_fusion_start_1ind,
    a.head_fusion_end_1ind,
    a.tail_fusion_start_1ind,
    a.tail_fusion_end_1ind,
    a.total_head_mutations,
    a.total_tail_mutations,
    a.all_valid
FROM pairings p
JOIN fusions f ON f.fusion_row_id = p.fusion_row_id
JOIN uniprot h ON h.uniprot_row_id = p.head_uniprot_row_id
JOIN uniprot t ON t.uniprot_row_id = p.tail_uniprot_row_id
LEFT JOIN alignments a ON a.pairing_id = p.pairing_id
"""


def _search_clause(
    *,
    query=None,
    fusiongenes=None,
    head_gene=None,
    tail_gene=None,
    seq_id=None,
    reviewed_only=False,
):
    clauses = []
    params = []
    text = str(query).strip() if query is not None else ""
    if text and not any([fusiongenes, head_gene, tail_gene, seq_id]):
        if "::" in text:
            fusiongenes = text
        elif text.lower().startswith("seq") and text[3:].isdigit():
            seq_id = "seq" + text[3:]
        else:
            head_gene = text
            tail_gene = text

    if fusiongenes:
        clauses.append("f.fusiongenes = ?")
        params.append(str(fusiongenes).strip().upper())
    if seq_id:
        token = str(seq_id).strip()
        if token.lower().startswith("seq") and token[3:].isdigit():
            token = "seq" + token[3:]
        clauses.append("f.seq_id = ?")
        params.append(token)
    gene_query = (
        head_gene
        and tail_gene
        and str(head_gene).strip().upper() == str(tail_gene).strip().upper()
        and not fusiongenes
    )
    if gene_query:
        gene = str(head_gene).strip().upper()
        clauses.append("(f.head_gene = ? OR f.tail_gene = ?)")
        params.extend([gene, gene])
    else:
        if head_gene:
            clauses.append("f.head_gene = ?")
            params.append(str(head_gene).strip().upper())
        if tail_gene:
            clauses.append("f.tail_gene = ?")
            params.append(str(tail_gene).strip().upper())
    if reviewed_only:
        clauses.append("p.both_reviewed = 1")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def search_pairings(
    conn: sqlite3.Connection,
    *,
    query=None,
    fusiongenes=None,
    head_gene=None,
    tail_gene=None,
    seq_id=None,
    reviewed_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Return one page of pairings and the total number of matches."""
    where, params = _search_clause(
        query=query,
        fusiongenes=fusiongenes,
        head_gene=head_gene,
        tail_gene=tail_gene,
        seq_id=seq_id,
        reviewed_only=reviewed_only,
    )
    total = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM pairings p
            JOIN fusions f ON f.fusion_row_id = p.fusion_row_id
            {where}
            """,
            params,
        ).fetchone()[0]
    )
    rows = conn.execute(
        _SEARCH_SELECT
        + where
        + " ORDER BY f.seq_id, h.uniprot_id, t.uniprot_id LIMIT ? OFFSET ?",
        [*params, int(limit), int(offset)],
    ).fetchall()
    frame = pd.DataFrame([dict(row) for row in rows])
    return frame, total


def compact_residue_maps(maps: dict) -> bytes:
    """Compress residue maps. Expanded again by ``expand_residue_maps``."""

    def pairs(partner):
        packed = []
        for entry in (maps.get(partner) or {}).values():
            packed.append(
                [
                    int(entry["parent_index_1ind"]),
                    int(entry["fusion_index_1ind"]),
                    entry["parent_aa"],
                    entry["fusion_aa"],
                ]
            )
        return packed

    def span(partner):
        raw = (maps.get("spans") or {}).get(partner)
        if not raw:
            return None
        return [
            raw.get("parent_start_1ind"),
            raw.get("parent_end_1ind"),
            raw.get("fusion_start_1ind"),
            raw.get("fusion_end_1ind"),
        ]

    payload = {
        "spans": {"head": span("head"), "tail": span("tail")},
        "head": pairs("head"),
        "tail": pairs("tail"),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return zlib.compress(raw, level=6)


def expand_residue_maps(blob: bytes) -> dict:
    payload = json.loads(zlib.decompress(blob))
    maps = {"head": {}, "tail": {}, "fusion": {}, "spans": {}}
    for partner in ("head", "tail"):
        span = (payload.get("spans") or {}).get(partner)
        if span:
            maps["spans"][partner] = {
                "parent_start_1ind": span[0],
                "parent_end_1ind": span[1],
                "fusion_start_1ind": span[2],
                "fusion_end_1ind": span[3],
            }
        for parent_i, fusion_i, parent_aa, fusion_aa in payload.get(partner) or []:
            entry = {
                "partner": partner,
                "parent_index_1ind": parent_i,
                "fusion_index_1ind": fusion_i,
                "parent_aa": parent_aa,
                "fusion_aa": fusion_aa,
                "is_mutation": parent_aa != fusion_aa,
            }
            maps[partner][str(parent_i)] = entry
            maps["fusion"].setdefault(str(fusion_i), entry)
    return maps


def get_pairing(conn: sqlite3.Connection, pairing_id: int) -> dict | None:
    """Load one pairing, its three sequences, and its alignment when present."""
    row = conn.execute(
        """
        SELECT
            p.pairing_id,
            f.fusion_row_id,
            f.seq_id,
            f.fusiongenes,
            f.head_gene,
            f.tail_gene,
            f.cancers,
            f.primary_sources,
            f.secondary_sources,
            h.uniprot_id AS head_uniprot_id,
            h.reviewed AS head_reviewed,
            h.isoform AS head_isoform,
            h.uniprot_id_full AS head_uniprot_id_full,
            h.entry_name AS head_entry_name,
            h.protein_names AS head_protein_names,
            t.uniprot_id AS tail_uniprot_id,
            t.reviewed AS tail_reviewed,
            t.isoform AS tail_isoform,
            t.uniprot_id_full AS tail_uniprot_id_full,
            t.entry_name AS tail_entry_name,
            t.protein_names AS tail_protein_names,
            p.both_reviewed,
            fs.sequence AS fusion_sequence,
            hs.sequence AS head_sequence,
            ts.sequence AS tail_sequence
        FROM pairings p
        JOIN fusions f ON f.fusion_row_id = p.fusion_row_id
        JOIN uniprot h ON h.uniprot_row_id = p.head_uniprot_row_id
        JOIN uniprot t ON t.uniprot_row_id = p.tail_uniprot_row_id
        JOIN sequences fs ON fs.sequence_id = f.sequence_id
        JOIN sequences hs ON hs.sequence_id = h.sequence_id
        JOIN sequences ts ON ts.sequence_id = t.sequence_id
        WHERE p.pairing_id = ?
        """,
        (int(pairing_id),),
    ).fetchone()
    if row is None:
        return None
    detail = dict(row)
    alignment = conn.execute(
        "SELECT * FROM alignments WHERE pairing_id = ?",
        (int(pairing_id),),
    ).fetchone()
    if alignment is None:
        detail["alignment"] = None
    else:
        stored = dict(alignment)
        blob = stored.pop("residue_maps", None)
        stored["residue_maps"] = expand_residue_maps(blob) if blob else None
        detail["alignment"] = stored
    return detail


def alignment_html(
    detail: dict,
    *,
    head_color: str = "crimson",
    tail_color: str = "royalblue",
    mutation_color: str = MUTATION_COLOR,
) -> str:
    """Color one stored alignment from its residue maps."""
    alignment = detail.get("alignment") or {}
    maps = alignment.get("residue_maps")
    if not maps:
        return "<p>No stored alignment for this pairing.</p>"

    def paint(sequence, partner, color):
        colored = {}
        mutations = set()
        for entry in (maps.get(partner) or {}).values():
            if partner == "fusion":
                continue
            index_key = "parent_index_1ind"
            idx = int(entry[index_key]) - 1
            colored[idx] = color
            if entry.get("is_mutation"):
                mutations.add(idx)
        return color_sequence_by_positions(
            sequence,
            colored,
            mutations,
            mutation_color=mutation_color,
        )

    fusion_colored = {}
    fusion_mutations = set()
    for partner, color in (("head", head_color), ("tail", tail_color)):
        for entry in (maps.get(partner) or {}).values():
            idx = int(entry["fusion_index_1ind"]) - 1
            fusion_colored[idx] = color
            if entry.get("is_mutation"):
                fusion_mutations.add(idx)
    fusion_html = color_sequence_by_positions(
        detail["fusion_sequence"],
        fusion_colored,
        fusion_mutations,
        mutation_color=mutation_color,
    )
    head_html = paint(detail["head_sequence"], "head", head_color)
    tail_html = paint(detail["tail_sequence"], "tail", tail_color)
    legend = sequence_color_legend_html(head_color, tail_color, mutation_color)
    head_label = f"{detail['head_gene']} ({detail['head_uniprot_id']})"
    tail_label = f"{detail['tail_gene']} ({detail['tail_uniprot_id']})"
    return f"""
    <div class="sequence-box">
      {legend}
      <b>{detail['fusiongenes']} · {detail['seq_id']}</b><br>
      {fusion_html}<br><br>
      <b>{head_label}</b><br>
      {head_html}<br><br>
      <b>{tail_label}</b><br>
      {tail_html}
    </div>
    """


def _alignment_row(pairing_id, fusion_seq, head_seq, tail_seq, raw_result):
    base = {
        "pairing_id": pairing_id,
        "status": "failed",
        "error": None,
        "score": None,
        "breakpoint_after_0ind": None,
        "head_fusion_start_1ind": None,
        "head_fusion_end_1ind": None,
        "tail_fusion_start_1ind": None,
        "tail_fusion_end_1ind": None,
        "head_parent_start_1ind": None,
        "head_parent_end_1ind": None,
        "tail_parent_start_1ind": None,
        "tail_parent_end_1ind": None,
        "head_fragment_length": None,
        "tail_fragment_length": None,
        "fusion_length": len(fusion_seq),
        "head_length": len(head_seq),
        "tail_length": len(tail_seq),
        "pct_fusion_from_head": None,
        "pct_fusion_from_tail": None,
        "pct_head_from_head_fragment": None,
        "pct_tail_from_tail_fragment": None,
        "total_head_mutations": None,
        "total_tail_mutations": None,
        "all_valid": None,
        "residue_maps": None,
    }
    if raw_result is None:
        base["error"] = "find_fusion_breakpoint returned None"
        return base

    result = normalize_breakpoint_result(raw_result, fusion_seq=fusion_seq)
    hhit = result["head_hit"]
    thit = result["tail_hit"]
    validation = validate_fragments(
        fusion_seq=fusion_seq,
        head_seq=head_seq,
        tail_seq=tail_seq,
        hfrag=result["hfrag"],
        tfrag=result["tfrag"],
    )
    h_fusion_start, h_fusion_end = validation["fusion_head_bounds"]
    t_fusion_start, t_fusion_end = validation["fusion_tail_bounds"]
    h_parent_start, h_parent_end = validation["head_parent_bounds"]
    t_parent_start, t_parent_end = validation["tail_parent_bounds"]
    head_mutations = mutations_from_hit(hhit)
    tail_mutations = mutations_from_hit(thit)
    hfrag_len = len(result["hfrag"])
    tfrag_len = len(result["tfrag"])
    maps = build_residue_maps(hhit, thit)
    base.update(
        {
            "status": "ok",
            "score": _sql_number(result.get("score")),
            "breakpoint_after_0ind": _sql_number(result.get("breakpoint_after")),
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
            "pct_fusion_from_head": _sql_number(pct(hfrag_len, len(fusion_seq))),
            "pct_fusion_from_tail": _sql_number(pct(tfrag_len, len(fusion_seq))),
            "pct_head_from_head_fragment": _sql_number(pct(hfrag_len, len(head_seq))),
            "pct_tail_from_tail_fragment": _sql_number(pct(tfrag_len, len(tail_seq))),
            "total_head_mutations": len(head_mutations),
            "total_tail_mutations": len(tail_mutations),
            "all_valid": 1 if validation["checks"]["all_valid"] else 0,
            "residue_maps": compact_residue_maps(maps),
        }
    )
    return base


_ALIGNMENT_INSERT = """
INSERT OR REPLACE INTO alignments (
    pairing_id, status, error, score, breakpoint_after_0ind,
    head_fusion_start_1ind, head_fusion_end_1ind,
    tail_fusion_start_1ind, tail_fusion_end_1ind,
    head_parent_start_1ind, head_parent_end_1ind,
    tail_parent_start_1ind, tail_parent_end_1ind,
    head_fragment_length, tail_fragment_length,
    fusion_length, head_length, tail_length,
    pct_fusion_from_head, pct_fusion_from_tail,
    pct_head_from_head_fragment, pct_tail_from_tail_fragment,
    total_head_mutations, total_tail_mutations, all_valid, residue_maps
) VALUES (
    :pairing_id, :status, :error, :score, :breakpoint_after_0ind,
    :head_fusion_start_1ind, :head_fusion_end_1ind,
    :tail_fusion_start_1ind, :tail_fusion_end_1ind,
    :head_parent_start_1ind, :head_parent_end_1ind,
    :tail_parent_start_1ind, :tail_parent_end_1ind,
    :head_fragment_length, :tail_fragment_length,
    :fusion_length, :head_length, :tail_length,
    :pct_fusion_from_head, :pct_fusion_from_tail,
    :pct_head_from_head_fragment, :pct_tail_from_tail_fragment,
    :total_head_mutations, :total_tail_mutations, :all_valid, :residue_maps
)
"""


def _parent_best_hits(sequence, fusion_seq, *, side, find_kwargs):
    kwargs = {
        "match": find_kwargs.get("match", 2),
        "mismatch": find_kwargs.get("mismatch", -1),
        "gap": find_kwargs.get("gap", -2),
    }
    if side == "head":
        hits = smith_waterman_hits_by_fusion_end(sequence, fusion_seq, **kwargs)
        return prefix_best_hits(hits)
    hits = suffix_best_hits_tail(sequence, fusion_seq, **kwargs)
    return hits


def align_fusion_row(
    conn: sqlite3.Connection,
    fusion_row_id: int,
    *,
    reviewed_only: bool = False,
    find_kwargs=None,
) -> dict:
    """Align every still-missing pairing of one oncoprotein and store the rows.

    Head Smith–Waterman is run once per distinct head sequence, and tail
    Smith–Waterman once per distinct tail sequence. Pairings reuse those hits.
    """
    find_kwargs = dict(find_kwargs or {})
    gap_penalty = int(find_kwargs.get("inter_fragment_gap_penalty", 1))
    fusion_seq = conn.execute(
        """
        SELECT s.sequence
        FROM fusions f
        JOIN sequences s ON s.sequence_id = f.sequence_id
        WHERE f.fusion_row_id = ?
        """,
        (int(fusion_row_id),),
    ).fetchone()
    if fusion_seq is None:
        raise KeyError(f"Unknown fusion_row_id {fusion_row_id}")
    fusion_seq = fusion_seq[0]

    reviewed_clause = "AND p.both_reviewed = 1" if reviewed_only else ""
    pending = conn.execute(
        f"""
        SELECT
            p.pairing_id,
            h.sequence_id AS head_sequence_id,
            t.sequence_id AS tail_sequence_id,
            hs.sequence AS head_sequence,
            ts.sequence AS tail_sequence
        FROM pairings p
        JOIN uniprot h ON h.uniprot_row_id = p.head_uniprot_row_id
        JOIN uniprot t ON t.uniprot_row_id = p.tail_uniprot_row_id
        JOIN sequences hs ON hs.sequence_id = h.sequence_id
        JOIN sequences ts ON ts.sequence_id = t.sequence_id
        LEFT JOIN alignments a ON a.pairing_id = p.pairing_id
        WHERE p.fusion_row_id = ?
          AND a.pairing_id IS NULL
          {reviewed_clause}
        """,
        (int(fusion_row_id),),
    ).fetchall()
    if not pending:
        return {"ok": 0, "failed": 0, "pairings": 0}

    head_hits = {}
    tail_hits = {}
    records = []
    n_ok = 0
    n_failed = 0
    for row in pending:
        head_id = row["head_sequence_id"]
        tail_id = row["tail_sequence_id"]
        if head_id not in head_hits:
            head_hits[head_id] = _parent_best_hits(
                row["head_sequence"], fusion_seq, side="head", find_kwargs=find_kwargs
            )
        if tail_id not in tail_hits:
            tail_hits[tail_id] = _parent_best_hits(
                row["tail_sequence"], fusion_seq, side="tail", find_kwargs=find_kwargs
            )
        try:
            raw = choose_breakpoint(
                head_hits[head_id],
                tail_hits[tail_id],
                inter_fragment_gap_penalty=gap_penalty,
            )
            record = _alignment_row(
                row["pairing_id"],
                fusion_seq,
                row["head_sequence"],
                row["tail_sequence"],
                raw,
            )
        except Exception as exc:
            record = _alignment_row(
                row["pairing_id"],
                fusion_seq,
                row["head_sequence"],
                row["tail_sequence"],
                None,
            )
            record["error"] = str(exc)
        if record["status"] == "ok":
            n_ok += 1
        else:
            n_failed += 1
        records.append(record)

    conn.executemany(_ALIGNMENT_INSERT, records)
    return {"ok": n_ok, "failed": n_failed, "pairings": len(records)}


def list_unaligned_fusion_ids(conn: sqlite3.Connection, *, reviewed_only: bool = False):
    reviewed_clause = "AND p.both_reviewed = 1" if reviewed_only else ""
    rows = conn.execute(
        f"""
        SELECT DISTINCT p.fusion_row_id
        FROM pairings p
        LEFT JOIN alignments a ON a.pairing_id = p.pairing_id
        WHERE a.pairing_id IS NULL
          {reviewed_clause}
        ORDER BY p.fusion_row_id
        """
    ).fetchall()
    return [int(row[0]) for row in rows]


def align_catalog(
    db_path=None,
    *,
    reviewed_only: bool = False,
    limit_fusions: int | None = None,
    find_kwargs=None,
) -> dict:
    """Run breakpoint analysis for pairings that do not yet have a stored result."""
    conn = open_catalog(db_path)
    try:
        fusion_ids = list_unaligned_fusion_ids(conn, reviewed_only=reviewed_only)
        if limit_fusions is not None:
            fusion_ids = fusion_ids[: int(limit_fusions)]
        totals = {"fusions": len(fusion_ids), "ok": 0, "failed": 0, "pairings": 0}
        started = time.time()
        for index, fusion_row_id in enumerate(fusion_ids, start=1):
            stats = align_fusion_row(
                conn,
                fusion_row_id,
                reviewed_only=reviewed_only,
                find_kwargs=find_kwargs,
            )
            conn.commit()
            totals["ok"] += stats["ok"]
            totals["failed"] += stats["failed"]
            totals["pairings"] += stats["pairings"]
            if index == 1 or index % 25 == 0 or index == len(fusion_ids):
                elapsed = time.time() - started
                print(
                    f"{index:,}/{len(fusion_ids):,} fusions, "
                    f"{totals['pairings']:,} pairings stored "
                    f"({totals['ok']:,} ok, {totals['failed']:,} failed) "
                    f"in {elapsed:.0f}s"
                )
        return totals
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build the FusOn-DB alignment catalog.")
    parser.add_argument("--db", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--id-map", default=str(DEFAULT_ID_MAP_CSV))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_catalog(args.db, id_map_csv=args.id_map, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
