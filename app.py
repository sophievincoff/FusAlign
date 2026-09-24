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


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
