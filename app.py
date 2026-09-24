import json
import re
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from fusion_breakpoint_pipeline import (
    format_residue_query_answer,
    query_residue,
    run_fusion_breakpoint_batch,
)
from fusondb_catalog import (
    alignment_html,
    default_catalog_path,
    get_pairing,
    open_catalog,
    search_pairings,
)


EXAMPLE_FUSION_NAME = "LASP1::RBX1"
EXAMPLE_FUSION_NAME_CSV = ("LASP1::RBX1", "APTX::ARL5B")

EXAMPLE_FUSION = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH"
)
EXAMPLE_FUSION_CSV = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH",
    "MSNVNLSVSDFWRVMMRVCWLVRQDSRHQRIRLPHLEAVVIGRGPETKITDKKCSRQQVQLKAECNKGYVKVKQVGVNPTSIDSVVIGKDQEVKLQPGQVLHMFIILVVDSIDRERLAITKEELYRMLAHEDLRKAAVLIFANKQDMKGCMTAAEISKYLTLSSIKDHPWHIQSCCALTGEGLCQGLEWMTSRIGVR",
)

EXAMPLE_HEAD = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQVRYKEEFEKNKGKGFSVVADTPELQRIKKTQDQISNIKYHEEFEKSRMGPSGGEGMEPERRDSQDGSSYRRPLEQQQPHHIPTSAPVYQQPQQQPVAQSYGGYKEPAAPVSIQRSAPGGGGKRYRAVYDYSAADEDEVSFQDGDTIVNVQQIDDGWMYGTVERTGDTGMLPANYVEAI"
)
EXAMPLE_HEAD_CSV = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQVRYKEEFEKNKGKGFSVVADTPELQRIKKTQDQISNIKYHEEFEKSRMGPSGGEGMEPERRDSQDGSSYRRPLEQQQPHHIPTSAPVYQQPQQQPVAQSYGGYKEPAAPVSIQRSAPGGGGKRYRAVYDYSAADEDEVSFQDGDTIVNVQQIDDGWMYGTVERTGDTGMLPANYVEAI",
    "MSNVNLSVSDFWRVMMRVCWLVRQDSRHQRIRLPHLEAVVIGRGPETKITDKKCSRQQVQLKAECNKGYVKVKQVGVNPTSIDSVVIGKDQEVKLQPGQVLHMVNELYPYIVEFEEEAKNPGLETHRKRKRSGNSDSIERDAAQEAEAGTGLEPGSNSGQCSVPLKKGKDAPIKKESLGHWSQGLKISMQDPKMQVYKDEQVVVIKDKYPKARYHWLVLPWTSISSLKAVAREHLELLKHMHTVGEKVIVDFAGSSKLRFRLGYHAIPSMSHVHLHVISQDFDSPCLKNKKHWNSFNTEYFLESQAVIEMVQEAGRVTVRDGMPELLKLPLRCHECQQLLPSIPQLKEHLRKHWTQ",
)

EXAMPLE_TAIL = (
    "MAAAMDVDTPSGTNSGAGKKRFEVKKWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH"
)
EXAMPLE_TAIL_CSV = (
    "MAAAMDVDTPSGTNSGAGKKRFEVKKWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH",
    "MGLIFAKLWSLFCNQEHKVIIVGLDNAGKTTILYQFLMNEVVHTSPTIGSNVEEIVVKNTHFLMWDIGGQESLRSSWNTYYSNTEFIILVVDSIDRERLAITKEELYRMLAHEDLRKAAVLIFANKQDMKGCMTAAEISKYLTLSSIKDHPWHIQSCCALTGEGLCQGLEWMTSRIGVR",
)


def make_example_csv() -> str:
    """Create a small example CSV file for download/upload testing."""
    example_dir = Path("examples")
    example_dir.mkdir(exist_ok=True)

    example_path = example_dir / "example_fusion_breakpoint_input.csv"

    df = pd.DataFrame({
        "fusion_name": list(EXAMPLE_FUSION_NAME_CSV),
        "fusion": list(EXAMPLE_FUSION_CSV),
        "head": list(EXAMPLE_HEAD_CSV),
        "tail": list(EXAMPLE_TAIL_CSV),
    })

    df.to_csv(example_path, index=False)
    return str(example_path)


EXAMPLE_CSV_PATH = make_example_csv()


def _read_uploaded_csv(csv_file):
    if csv_file is None:
        return None

    csv_path = csv_file if isinstance(csv_file, str) else csv_file.name
    return pd.read_csv(csv_path)


def _clean_sequence(seq):
    return str(seq).strip().replace("\n", "").replace(" ", "").replace("\t", "")


def _make_single_input_df(fusion_name, fusion_seq, head_seq, tail_seq):
    fusion_name = str(fusion_name).strip() or "fusion_1"
    fusion_seq = _clean_sequence(fusion_seq)
    head_seq = _clean_sequence(head_seq)
    tail_seq = _clean_sequence(tail_seq)

    if not fusion_seq or not head_seq or not tail_seq:
        raise gr.Error(
            "Please provide fusion, head, and tail sequences, or upload a CSV."
        )

    return pd.DataFrame({
        "fusion_name": [fusion_name],
        "fusion": [fusion_seq],
        "head": [head_seq],
        "tail": [tail_seq],
    })


def _maps_by_fusion_name(results_df: pd.DataFrame) -> dict:
    maps = {}
    if results_df is None or results_df.empty:
        return maps
    if "residue_maps_json" not in results_df.columns:
        return maps

    for _, row in results_df.iterrows():
        if row.get("status") != "ok":
            continue
        raw = row.get("residue_maps_json")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            maps[str(row["fusion_name"])] = (
                json.loads(raw) if isinstance(raw, str) else raw
            )
        except (TypeError, json.JSONDecodeError):
            continue
    return maps


def _fusion_base_name(name: str) -> str:
    """Drop duplicate tags added for display (e.g., _seq2)."""
    return re.sub(r"_seq\d+$", "", str(name))


def _partner_aliases_from_fusion_name(fusion_name: str) -> dict:
    """
    Build partner aliases from fusion labels like Head::Tail or Head-Tail.
    """
    aliases = {}
    base = _fusion_base_name(fusion_name).strip()
    if "::" in base:
        parts = base.split("::", 1)
    elif "-" in base:
        parts = base.split("-", 1)
    else:
        return aliases

    if len(parts) != 2:
        return aliases

    head_name = parts[0].strip()
    tail_name = parts[1].strip()
    if head_name:
        aliases[head_name.lower()] = "head"
    if tail_name:
        aliases[tail_name.lower()] = "tail"
    return aliases


def _input_signature(input_df: pd.DataFrame, fusion_col, head_col, tail_col, fusion_name_col) -> str:
    """Stable signature to detect same biological input regardless of colors."""
    keep = [fusion_name_col, fusion_col, head_col, tail_col]
    safe_df = input_df[keep].astype(str).copy()
    return safe_df.to_csv(index=False)


def _restyle_html_colors(html_text: str, old_colors: dict, new_colors: dict) -> str:
    """Fast color-only refresh by replacing previous color literals in generated HTML."""
    if not old_colors:
        return html_text
    out = html_text
    for key in ("head", "tail", "mutation"):
        old = old_colors.get(key)
        new = new_colors.get(key)
        if old and new and old != new:
            out = out.replace(str(old), str(new))
    return out


def load_single_example():
    return EXAMPLE_FUSION_NAME, EXAMPLE_FUSION, EXAMPLE_HEAD, EXAMPLE_TAIL


def _pairing_choice_label(row) -> str:
    status = row.alignment_status if pd.notna(row.alignment_status) else "not aligned"
    return (
        f"{int(row.pairing_id)} | {row.seq_id} | "
        f"{row.head_uniprot_id} × {row.tail_uniprot_id} | {status}"
    )


def search_known_fusions(query, reviewed_only):
    empty = pd.DataFrame()
    path = default_catalog_path()
    if not path.exists():
        return (
            f"Catalog not built yet. Expected `{path}`.",
            empty,
            gr.update(choices=[], value=None),
        )
    text = str(query or "").strip()
    if not text:
        return (
            "Enter a fusion (`A1BG::FGA`), a gene symbol, or a FusOn-DB id (`seq1`).",
            empty,
            gr.update(choices=[], value=None),
        )

    conn = open_catalog(path)
    try:
        found, total = search_pairings(
            conn,
            query=text,
            reviewed_only=bool(reviewed_only),
            limit=100,
        )
    finally:
        conn.close()

    if found.empty:
        return (
            f"No pairings for `{text}`.",
            empty,
            gr.update(choices=[], value=None),
        )

    choices = [_pairing_choice_label(row) for row in found.itertuples(index=False)]
    summary = (
        f"**{total:,}** pairings for `{text}`. "
        f"Showing {len(found):,}."
    )
    return summary, found, gr.update(choices=choices, value=choices[0])


def inspect_known_pairing(choice, residue_query):
    if not choice:
        return "Select a pairing from the search results.", ""
    pairing_id = int(str(choice).split("|", 1)[0].strip())
    conn = open_catalog()
    try:
        detail = get_pairing(conn, pairing_id)
    finally:
        conn.close()
    if detail is None:
        return f"Pairing `{pairing_id}` was not found.", ""

    alignment = detail.get("alignment")
    html = alignment_html(detail) if alignment and alignment.get("residue_maps") else ""
    header = (
        f"**{detail['fusiongenes']}** ({detail['seq_id']}) · "
        f"head `{detail['head_uniprot_id']}` ({detail['head_reviewed']}) × "
        f"tail `{detail['tail_uniprot_id']}` ({detail['tail_reviewed']})"
    )
    if alignment is None:
        return (
            header
            + "\n\nThis pairing is in the catalog. Its breakpoint alignment has not been stored yet.",
            "",
        )
    if alignment.get("status") != "ok" or not alignment.get("residue_maps"):
        return header + f"\n\nAlignment failed: {alignment.get('error')}", ""

    if not residue_query or not str(residue_query).strip():
        return (
            header
            + f"\n\n- Score: `{alignment.get('score')}`"
            + f"\n- Head in fusion: `{alignment.get('head_fusion_start_1ind')}–{alignment.get('head_fusion_end_1ind')}`"
            + f"\n- Tail in fusion: `{alignment.get('tail_fusion_start_1ind')}–{alignment.get('tail_fusion_end_1ind')}`"
            + f"\n- Mutations: head `{alignment.get('total_head_mutations')}`, "
            + f"tail `{alignment.get('total_tail_mutations')}`",
            html,
        )

    aliases = _partner_aliases_from_fusion_name(detail["fusiongenes"])
    answer = format_residue_query_answer(
        query_residue(
            alignment["residue_maps"],
            query_text=residue_query,
            partner_aliases=aliases,
        )
    )
    return header + "\n\n" + answer, html


def run_app(
    fusion_name,
    fusion_seq,
    head_seq,
    tail_seq,
    csv_file,
    fusion_col,
    head_col,
    tail_col,
    fusion_name_col,
    head_color,
    tail_color,
    mutation_color,
    run_cache,
):
    out_dir = Path(tempfile.mkdtemp(prefix="fusion_breakpoint_gradio_"))

    uploaded_df = _read_uploaded_csv(csv_file)

    if uploaded_df is not None:
        input_df = uploaded_df
    else:
        input_df = _make_single_input_df(
            fusion_name=fusion_name,
            fusion_seq=fusion_seq,
            head_seq=head_seq,
            tail_seq=tail_seq,
        )

    signature = _input_signature(
        input_df,
        fusion_col=fusion_col,
        head_col=head_col,
        tail_col=tail_col,
        fusion_name_col=fusion_name_col,
    )
    new_colors = {
        "head": head_color,
        "tail": tail_color,
        "mutation": mutation_color,
    }

    can_fast_refresh = (
        isinstance(run_cache, dict)
        and run_cache.get("signature") == signature
        and run_cache.get("html_path")
        and run_cache.get("csv_path")
        and run_cache.get("results_records") is not None
    )

    if can_fast_refresh:
        results_df = pd.DataFrame(run_cache["results_records"])
        base_html = Path(run_cache["html_path"]).read_text(encoding="utf-8")
        html_text = _restyle_html_colors(
            base_html,
            old_colors=run_cache.get("colors", {}),
            new_colors=new_colors,
        )
        html_path = out_dir / "fusion_breakpoint_report.html"
        html_path.write_text(html_text, encoding="utf-8")
        csv_path = out_dir / "fusion_breakpoint_summary.csv"
        csv_path.write_text(
            Path(run_cache["csv_path"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    else:
        results_df, html_path, csv_path = run_fusion_breakpoint_batch(
            input_df,
            fusion_col=fusion_col,
            head_col=head_col,
            tail_col=tail_col,
            fusion_name_col=fusion_name_col,
            out_dir=out_dir,
            combined_html_name="fusion_breakpoint_report.html",
            combined_csv_name="fusion_breakpoint_summary.csv",
            write_per_fusion_html=True,
            display_combined=False,
            head_color=head_color,
            tail_color=tail_color,
            mutation_color=mutation_color,
        )
        html_text = Path(html_path).read_text(encoding="utf-8")

    preview_rename = {
        "fusion_name": "Fusion",
        "status": "Status",
        "fusion_length": "Length",
        "head_fusion_start_1ind": "Head start",
        "head_fusion_end_1ind": "Head end",
        "tail_fusion_start_1ind": "Tail start",
        "tail_fusion_end_1ind": "Tail end",
        "pct_fusion_from_head": "% head",
        "pct_fusion_from_tail": "% tail",
        "total_head_mutations": "Head muts",
        "total_tail_mutations": "Tail muts",
        "error": "Error",
    }
    preview_cols = [c for c in preview_rename if c in results_df.columns]
    preview_df = results_df[preview_cols].rename(columns=preview_rename)

    maps = _maps_by_fusion_name(results_df)
    ok_names = list(maps.keys())
    fusion_dropdown = gr.update(
        choices=ok_names,
        value=ok_names[0] if ok_names else None,
    )

    return (
        preview_df,
        html_text,
        str(html_path),
        str(csv_path),
        maps,
        fusion_dropdown,
        "",
        {
            "signature": signature,
            "html_path": str(html_path),
            "csv_path": str(csv_path),
            "colors": new_colors,
            "results_records": results_df.to_dict(orient="records"),
        },
    )


def run_residue_query(maps_by_name, fusion_name, query_text):
    if not maps_by_name:
        return "Run **Generate report** first, then query a residue."
    if not fusion_name:
        return "Select a fusion from the dropdown."
    if fusion_name not in maps_by_name:
        return f"No residue maps for `{fusion_name}`."
    if not query_text or not str(query_text).strip():
        return (
            "Enter a query such as `Tail:Y1078`, `Tail:1078-1085`, "
            "`Head:S12`, `Fusion:Y200`, or partner alias like `ALK:Y1078`."
        )

    partner_aliases = _partner_aliases_from_fusion_name(fusion_name)
    answer = query_residue(
        maps_by_name[fusion_name],
        query_text=query_text,
        partner_aliases=partner_aliases,
    )
    return format_residue_query_answer(answer)


with gr.Blocks(
    title="Fusion Breakpoint Reporter",
    theme=gr.themes.Soft(),
    css="""
    #title-block {
        text-align: center;
        margin-bottom: 1.25rem;
    }
    #title-block h1 {
        font-size: 2.2rem;
        margin-bottom: 0.25rem;
    }
    #title-block p {
        color: #475569;
        font-size: 1rem;
    }
    #summary-preview {
        overflow-x: auto;
    }
    #summary-preview table {
        table-layout: auto;
        width: max-content;
        min-width: 100%;
        border-collapse: collapse;
    }
    #summary-preview th,
    #summary-preview td {
        white-space: nowrap !important;
        padding: 10px 14px !important;
        font-size: 13px !important;
        line-height: 1.35 !important;
        vertical-align: middle !important;
    }
    #summary-preview th {
        font-weight: 700 !important;
        letter-spacing: 0.01em;
    }
    """
) as demo:
    residue_maps_state = gr.State({})
    run_cache_state = gr.State({})

    gr.HTML(
        """
        <div id="title-block">
          <h1>Fusion Breakpoint Reporter</h1>
          <p>
            Identify head/tail breakpoint regions in fusion oncoproteins,
            generate a polished HTML report, and download summary results.
          </p>
        </div>
        """
    )

    with gr.Tab("Single fusion"):
        load_example_button = gr.Button("Load LASP1::RBX1 example")

        fusion_name = gr.Textbox(
            label="Fusion name",
            value="",
            placeholder="e.g. LASP1::RBX1",
        )

        fusion_seq = gr.Textbox(
            label="Fusion oncoprotein sequence",
            lines=6,
            placeholder="Paste the full fusion sequence here...",
        )

        head_seq = gr.Textbox(
            label="Head parent protein sequence",
            lines=6,
            placeholder="Paste the head parent sequence here...",
        )

        tail_seq = gr.Textbox(
            label="Tail parent protein sequence",
            lines=6,
            placeholder="Paste the tail parent sequence here...",
        )

    with gr.Tab("Batch CSV upload"):
        gr.Markdown(
            """
            Upload a CSV with columns for fusion name, fusion sequence,
            head sequence, and tail sequence.

            Default expected columns: `fusion_name`, `fusion`, `head`, `tail`.
            """
        )

        example_csv_download = gr.File(
            label="Download example CSV",
            value=EXAMPLE_CSV_PATH,
            interactive=False,
        )

        csv_file = gr.File(
            label="Upload input CSV",
            file_types=[".csv"],
        )

        with gr.Row():
            fusion_name_col = gr.Textbox(label="Fusion name column", value="fusion_name")
            fusion_col = gr.Textbox(label="Fusion sequence column", value="fusion")
            head_col = gr.Textbox(label="Head sequence column", value="head")
            tail_col = gr.Textbox(label="Tail sequence column", value="tail")

    with gr.Tab("Known fusions"):
        gr.Markdown(
            """
            Search the FusOn-DB catalog. Each row is one fusion oncoprotein
            paired with one head UniProt entry and one tail UniProt entry.
            Reviewed-only is the priority set: both parents are Swiss-Prot reviewed.
            Breakpoint coordinates and residue queries appear after that pairing
            has been aligned and stored.
            """
        )
        with gr.Row():
            catalog_query = gr.Textbox(
                label="Fusion, gene, or seq id",
                placeholder="A1BG::FGA",
                scale=3,
            )
            catalog_reviewed = gr.Checkbox(
                label="Reviewed head and tail only",
                value=True,
            )
            catalog_search_button = gr.Button("Search catalog", variant="primary")
        catalog_summary = gr.Markdown()
        catalog_table = gr.Dataframe(
            label="Matching pairings",
            interactive=False,
            wrap=False,
        )
        catalog_choice = gr.Dropdown(
            label="Open a pairing",
            choices=[],
            value=None,
            interactive=True,
        )
        with gr.Row():
            catalog_residue = gr.Textbox(
                label="Residue query",
                placeholder="Head:M1",
                scale=3,
            )
            catalog_residue_button = gr.Button("Show alignment", variant="secondary")
        catalog_answer = gr.Markdown()
        catalog_html = gr.HTML()

    with gr.Accordion("Display options", open=False):
        with gr.Row():
            head_color = gr.ColorPicker(label="Head color", value="#dc143c")
            tail_color = gr.ColorPicker(label="Tail color", value="#4169e1")
            mutation_color = gr.ColorPicker(label="Mutation color", value="#7c3aed")

    with gr.Row():
        run_button = gr.Button("Generate report", variant="primary")
        cancel_button = gr.Button("Cancel job", variant="stop")
        refresh_button = gr.Button("Refresh session")

    gr.Markdown("## Results")

    results_preview = gr.Dataframe(
        label="Summary preview",
        interactive=False,
        wrap=False,
        elem_id="summary-preview",
    )

    html_preview = gr.HTML(label="HTML report preview")

    with gr.Row():
        html_download = gr.File(label="Download combined HTML report")
        csv_download = gr.File(label="Download summary CSV")

    gr.Markdown(
        """
        ## Residue query
        After a successful run, map a parent or fusion residue without re-aligning.
        Syntax examples:
        `Tail:Y1078`, `Tail:1078-1085`,
        `Head:S12`, `Fusion:Y200`, or aliases like `ALK:Y1078` when the fusion name
        is `Head::Tail` or `Head-Tail`.
        """
    )
    with gr.Row():
        query_fusion = gr.Dropdown(
            label="Fusion",
            choices=[],
            value=None,
            interactive=True,
        )
        query_text = gr.Textbox(
            label="Residue query",
            placeholder="Tail:Y1078",
            scale=2,
        )
        query_button = gr.Button("Query", variant="secondary")

    query_answer = gr.Markdown(value="")

    load_example_button.click(
        fn=load_single_example,
        inputs=[],
        outputs=[
            fusion_name,
            fusion_seq,
            head_seq,
            tail_seq,
        ],
    )

    run_event = run_button.click(
        fn=run_app,
        inputs=[
            fusion_name,
            fusion_seq,
            head_seq,
            tail_seq,
            csv_file,
            fusion_col,
            head_col,
            tail_col,
            fusion_name_col,
            head_color,
            tail_color,
            mutation_color,
            run_cache_state,
        ],
        outputs=[
            results_preview,
            html_preview,
            html_download,
            csv_download,
            residue_maps_state,
            query_fusion,
            query_answer,
            run_cache_state,
        ],
    )

    query_button.click(
        fn=run_residue_query,
        inputs=[residue_maps_state, query_fusion, query_text],
        outputs=[query_answer],
    )
    query_text.submit(
        fn=run_residue_query,
        inputs=[residue_maps_state, query_fusion, query_text],
        outputs=[query_answer],
    )

    cancel_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[run_event],
    )

    refresh_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { window.location.reload(); }",
    )

    catalog_search_button.click(
        fn=search_known_fusions,
        inputs=[catalog_query, catalog_reviewed],
        outputs=[catalog_summary, catalog_table, catalog_choice],
    )
    catalog_query.submit(
        fn=search_known_fusions,
        inputs=[catalog_query, catalog_reviewed],
        outputs=[catalog_summary, catalog_table, catalog_choice],
    )
    catalog_residue_button.click(
        fn=inspect_known_pairing,
        inputs=[catalog_choice, catalog_residue],
        outputs=[catalog_answer, catalog_html],
    )
    catalog_choice.change(
        fn=inspect_known_pairing,
        inputs=[catalog_choice, catalog_residue],
        outputs=[catalog_answer, catalog_html],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
