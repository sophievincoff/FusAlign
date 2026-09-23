import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from fusion_breakpoint_pipeline import run_fusion_breakpoint_batch


EXAMPLE_FUSION_NAME = "LASP1::RBX1"
EXAMPLE_FUSION_NAME_CSV = ("LASP1::RBX1","APTX::ARL5B")

EXAMPLE_FUSION = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH"
)
EXAMPLE_FUSION_CSV = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH",
    "MSNVNLSVSDFWRVMMRVCWLVRQDSRHQRIRLPHLEAVVIGRGPETKITDKKCSRQQVQLKAECNKGYVKVKQVGVNPTSIDSVVIGKDQEVKLQPGQVLHMFIILVVDSIDRERLAITKEELYRMLAHEDLRKAAVLIFANKQDMKGCMTAAEISKYLTLSSIKDHPWHIQSCCALTGEGLCQGLEWMTSRIGVR"
)

EXAMPLE_HEAD = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQVRYKEEFEKNKGKGFSVVADTPELQRIKKTQDQISNIKYHEEFEKSRMGPSGGEGMEPERRDSQDGSSYRRPLEQQQPHHIPTSAPVYQQPQQQPVAQSYGGYKEPAAPVSIQRSAPGGGGKRYRAVYDYSAADEDEVSFQDGDTIVNVQQIDDGWMYGTVERTGDTGMLPANYVEAI"
)
EXAMPLE_HEAD_CSV = (
    "MNPNCARCGKIVYPTEKVNCLDKFWHKACFHCETCKMTLNMKNYKGYEKKPYCNAHYPKQSFTMVADTPENLRLKQQSELQSQVRYKEEFEKNKGKGFSVVADTPELQRIKKTQDQISNIKYHEEFEKSRMGPSGGEGMEPERRDSQDGSSYRRPLEQQQPHHIPTSAPVYQQPQQQPVAQSYGGYKEPAAPVSIQRSAPGGGGKRYRAVYDYSAADEDEVSFQDGDTIVNVQQIDDGWMYGTVERTGDTGMLPANYVEAI",
    "MSNVNLSVSDFWRVMMRVCWLVRQDSRHQRIRLPHLEAVVIGRGPETKITDKKCSRQQVQLKAECNKGYVKVKQVGVNPTSIDSVVIGKDQEVKLQPGQVLHMVNELYPYIVEFEEEAKNPGLETHRKRKRSGNSDSIERDAAQEAEAGTGLEPGSNSGQCSVPLKKGKDAPIKKESLGHWSQGLKISMQDPKMQVYKDEQVVVIKDKYPKARYHWLVLPWTSISSLKAVAREHLELLKHMHTVGEKVIVDFAGSSKLRFRLGYHAIPSMSHVHLHVISQDFDSPCLKNKKHWNSFNTEYFLESQAVIEMVQEAGRVTVRDGMPELLKLPLRCHECQQLLPSIPQLKEHLRKHWTQ"
)

EXAMPLE_TAIL = (
    "MAAAMDVDTPSGTNSGAGKKRFEVKKWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH"
)
EXAMPLE_TAIL_CSV = (
    "MAAAMDVDTPSGTNSGAGKKRFEVKKWNAVALWAWDIVVDNCAICRNHIMDLCIECQANQASATSEECTVAWGVCNHAFHFHCISRWLKTRQVCPLDNREWEFQKYGH",
    "MGLIFAKLWSLFCNQEHKVIIVGLDNAGKTTILYQFLMNEVVHTSPTIGSNVEEIVVKNTHFLMWDIGGQESLRSSWNTYYSNTEFIILVVDSIDRERLAITKEELYRMLAHEDLRKAAVLIFANKQDMKGCMTAAEISKYLTLSSIKDHPWHIQSCCALTGEGLCQGLEWMTSRIGVR"
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
    )

    html_text = Path(html_path).read_text(encoding="utf-8")

    preview_cols = [
        "fusion_name",
        "status",
        "fusion_length",
        "head_fusion_start_1ind",
        "head_fusion_end_1ind",
        "tail_fusion_start_1ind",
        "tail_fusion_end_1ind",
        "pct_fusion_from_head",
        "pct_fusion_from_tail",
        "total_head_mutations",
        "total_tail_mutations",
        "error",
    ]
    preview_cols = [c for c in preview_cols if c in results_df.columns]

    return results_df[preview_cols].copy(), html_text, html_path, csv_path


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
    """
) as demo:
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
            head_color = gr.Textbox(label="Head color", value="crimson")
            tail_color = gr.Textbox(label="Tail color", value="royalblue")

    with gr.Row():
        run_button = gr.Button("Generate report", variant="primary")
        cancel_button = gr.Button("Cancel job", variant="stop")
        refresh_button = gr.Button("Refresh session")

    gr.Markdown("## Results")

    results_preview = gr.Dataframe(
        label="Summary preview",
        interactive=False,
        wrap=True,
    )

    html_preview = gr.HTML(label="HTML report preview")

    with gr.Row():
        html_download = gr.File(label="Download combined HTML report")
        csv_download = gr.File(label="Download summary CSV")

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
        ],
        outputs=[
            results_preview,
            html_preview,
            html_download,
            csv_download,
        ],
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