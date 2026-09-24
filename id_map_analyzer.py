"""Combine UniProt ID-mapping TSVs with FASTA isoform sequences."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

try:
    from Bio import SeqIO
except ImportError:  # pragma: no cover
    SeqIO = None


PathLike = Union[str, Path]

TSV_COLUMNS = [
    "From",
    "Entry",
    "Reviewed",
    "Entry Name",
    "Protein names",
    "Gene Names",
    "Organism",
    "Length",
]

OUTPUT_COLUMNS = [
    "Query",
    "UniProtID",
    "Reviewed",
    "Entry Name",
    "Protein names",
    "Gene Names",
    "Organism",
    "Length",
    "Sequence",
    "Isoform",
    "UniProtID_Full",
]

_ACCESSION_RE = re.compile(
    r"^(?P<base>[A-Za-z0-9]+)(?:-(?P<isoform>\d+))?$"
)
_HEADER_ACCESSION_RE = re.compile(
    r"^>(?:sp|tr)\|(?P<acc>[A-Za-z0-9]+(?:-\d+)?)\|"
)


def _as_path_list(paths: Optional[Union[PathLike, Iterable[PathLike]]]) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(p) for p in paths]


def parse_uniprot_accession(accession: str) -> tuple[str, int]:
    """
    Split a UniProt accession into (base_id, isoform_number).

    Canonical accessions (no ``-N`` suffix) map to isoform ``0``.
    """
    text = str(accession).strip()
    match = _ACCESSION_RE.match(text)
    if not match:
        raise ValueError(f"Unrecognized UniProt accession: {accession!r}")
    base = match.group("base")
    isoform = int(match.group("isoform") or 0)
    return base, isoform


def uniprot_id_full(base_id: str, isoform: int) -> str:
    """Build ``UniProtID_Full`` as ``BASE-0`` or ``BASE-N``."""
    return f"{base_id}-{int(isoform)}"


class IDMapAnalyzer:
    """
    Merge one or more UniProt ID-mapping TSVs with one or more UniProt FASTAs.

    Output grain is **one row per isoform sequence**. TSV metadata is attached
    by matching the FASTA accession base ID to the TSV ``Entry`` column.
    """

    def __init__(
        self,
        tsv_paths: Optional[Union[PathLike, Iterable[PathLike]]] = None,
        fasta_paths: Optional[Union[PathLike, Iterable[PathLike]]] = None,
    ):
        self.tsv_paths = _as_path_list(tsv_paths)
        self.fasta_paths = _as_path_list(fasta_paths)
        self._df: Optional[pd.DataFrame] = None

    def add_tsv(self, path: PathLike) -> "IDMapAnalyzer":
        self.tsv_paths.append(Path(path))
        self._df = None
        return self

    def add_fasta(self, path: PathLike) -> "IDMapAnalyzer":
        self.fasta_paths.append(Path(path))
        self._df = None
        return self

    def load_tsvs(self) -> pd.DataFrame:
        """Concatenate all TSVs and rename From→Query, Entry→UniProtID."""
        if not self.tsv_paths:
            raise ValueError("No TSV paths provided.")

        frames = []
        for path in self.tsv_paths:
            frame = pd.read_csv(path, sep="\t", dtype=str)
            missing = [c for c in TSV_COLUMNS if c not in frame.columns]
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            frames.append(frame[TSV_COLUMNS].copy())

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.rename(columns={"From": "Query", "Entry": "UniProtID"})
        return combined

    def load_fastas(self) -> dict[str, list[dict]]:
        """
        Parse all FASTAs into ``base_id -> [{isoform, accession, sequence}, ...]``.
        """
        by_base: dict[str, list[dict]] = {}
        if not self.fasta_paths:
            return by_base

        for path in self.fasta_paths:
            for accession, sequence in self._iter_fasta(path):
                base, isoform = parse_uniprot_accession(accession)
                by_base.setdefault(base, []).append({
                    "isoform": isoform,
                    "accession": accession,
                    "sequence": sequence,
                })
        return by_base

    @staticmethod
    def _iter_fasta(path: Path):
        """Yield ``(accession, sequence)`` pairs from a UniProt FASTA."""
        if SeqIO is not None:
            for record in SeqIO.parse(str(path), "fasta"):
                accession = record.id.split("|")[1] if "|" in record.id else record.id
                yield accession, str(record.seq).replace(" ", "").replace("\n", "")
            return

        # Lightweight fallback if Biopython is unavailable.
        accession = None
        chunks: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if accession is not None:
                        yield accession, "".join(chunks)
                    match = _HEADER_ACCESSION_RE.match(line)
                    if not match:
                        raise ValueError(f"Unrecognized FASTA header in {path}: {line}")
                    accession = match.group("acc")
                    chunks = []
                else:
                    chunks.append(line)
            if accession is not None:
                yield accession, "".join(chunks)

    def build(self, *, force: bool = False) -> pd.DataFrame:
        """
        Build the combined table.

        Raises ``ValueError`` if the same ``UniProtID_Full`` appears with more
        than one distinct non-empty sequence.
        """
        if self._df is not None and not force:
            return self._df.copy()

        tsv = self.load_tsvs()
        fasta_by_base = self.load_fastas()

        rows = []
        for record in tsv.to_dict(orient="records"):
            base_id = str(record["UniProtID"]).strip()
            hits = fasta_by_base.get(base_id, [])

            if not hits:
                rows.append(self._make_row(record, isoform=0, sequence=""))
                continue

            # Stable order: isoform number, then accession.
            hits = sorted(hits, key=lambda h: (h["isoform"], h["accession"]))
            for hit in hits:
                rows.append(
                    self._make_row(
                        record,
                        isoform=hit["isoform"],
                        sequence=hit["sequence"],
                    )
                )

        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        df = self._dedupe_and_validate(df)
        self._df = df
        return df.copy()

    @staticmethod
    def _make_row(tsv_record: dict, *, isoform: int, sequence: str) -> dict:
        base_id = str(tsv_record["UniProtID"]).strip()
        sequence = sequence or ""
        length = len(sequence) if sequence else tsv_record.get("Length")
        return {
            "Query": tsv_record.get("Query"),
            "UniProtID": base_id,
            "Reviewed": tsv_record.get("Reviewed"),
            "Entry Name": tsv_record.get("Entry Name"),
            "Protein names": tsv_record.get("Protein names"),
            "Gene Names": tsv_record.get("Gene Names"),
            "Organism": tsv_record.get("Organism"),
            "Length": length,
            "Sequence": sequence,
            "Isoform": int(isoform),
            "UniProtID_Full": uniprot_id_full(base_id, isoform),
        }

    @staticmethod
    def _dedupe_and_validate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce unique ``UniProtID_Full`` with a single sequence.

        Identical ``(UniProtID_Full, Sequence)`` duplicates are collapsed.
        Conflicting sequences for the same ``UniProtID_Full`` raise.
        """
        if df.empty:
            return df

        # Empty sequences are allowed placeholders for TSV-only entries.
        nonempty = df[df["Sequence"].fillna("").astype(str) != ""].copy()
        if not nonempty.empty:
            conflict = (
                nonempty.groupby("UniProtID_Full")["Sequence"]
                .nunique(dropna=False)
                .reset_index(name="n_seq")
            )
            bad = conflict.loc[conflict["n_seq"] > 1, "UniProtID_Full"].tolist()
            if bad:
                examples = ", ".join(bad[:10])
                more = "" if len(bad) <= 10 else f" (+{len(bad) - 10} more)"
                raise ValueError(
                    "Conflicting sequences for UniProtID_Full: "
                    f"{examples}{more}"
                )

        # Prefer rows that have a sequence when collapsing duplicates.
        ordered = df.copy()
        ordered["_has_seq"] = ordered["Sequence"].fillna("").astype(str).ne("")
        ordered = ordered.sort_values(
            by=["UniProtID_Full", "_has_seq"],
            ascending=[True, False],
        )
        ordered = ordered.drop_duplicates(subset=["UniProtID_Full"], keep="first")
        ordered = ordered.drop(columns=["_has_seq"])
        ordered = ordered.reset_index(drop=True)
        return ordered[OUTPUT_COLUMNS]

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for :meth:`build`."""
        return self.build()

    def to_csv(self, path: PathLike, **kwargs) -> Path:
        """Write the combined table to CSV and return the output path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.build()
        write_kwargs = {"index": False}
        write_kwargs.update(kwargs)
        df.to_csv(out, **write_kwargs)
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine UniProt ID-mapping TSVs and FASTAs into one CSV."
    )
    parser.add_argument("--tsv", nargs="+", required=True, help="One or more TSV paths")
    parser.add_argument("--fasta", nargs="+", default=[], help="One or more FASTA paths")
    parser.add_argument(
        "-o",
        "--output",
        default="uniprot_id_map_combined.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    analyzer = IDMapAnalyzer(tsv_paths=args.tsv, fasta_paths=args.fasta)
    out_path = analyzer.to_csv(args.output)
    df = analyzer.to_dataframe()
    print(f"Wrote {len(df):,} rows to {out_path}")
