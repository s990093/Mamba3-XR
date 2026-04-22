#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
import shutil
import socketserver
import subprocess
import threading
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_MD = PROJECT_DIR / "report.md"
OUTPUT_PDF = PROJECT_DIR / "report.pdf"

DIAGRAM_SPECS = (
    ("prototypes/method_flowchart.html", "assets/method_flowchart.svg"),
    ("prototypes/architecture.html", "assets/images/architecture.svg"),
)

METHOD_PIPELINE_MD = "![Method Pipeline](./assets/method_flowchart.svg)"
METHOD_PIPELINE_LATEX = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{./assets/method_flowchart.png}
\caption{Method Pipeline}
\end{figure}
""".strip()

MODEL_ARCH_MD = "![Model Architecture](./assets/images/architecture.svg)"
MODEL_ARCH_CAPTION_MD = (
    "圖 2：Hybrid Mamba-TuckerMoE 詳細模型架構。每個 Macro Block 包含 4 個 "
    "Mamba3Block 與 1 個 TransformerBlock，整體堆疊 $N_\\text{macro}=6$ 次。"
)
MODEL_ARCH_LATEX_TEMPLATE = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{./assets/images/architecture.png}
\caption{Hybrid Mamba-TuckerMoE 詳細模型架構。每個 Macro Block 包含 4 個 Mamba3Block 與 1 個 TransformerBlock，整體堆疊 $N_\text{macro}=6$ 次。}
\end{figure}
""".strip()

APPENDIX_ALGO_LATEX = {
    "./assets/algorithms/appendix_a1_tuckermoe_forward.png": r"""
\begin{algorithm}[H]
\caption{TuckerMoE Forward Pass}
\begin{algorithmic}[1]
\Procedure{TuckerMoEForward}{$x, W_r, U_{\mathrm{in}}, U_{\mathrm{exp}}, \mathcal{G}, U_{\mathrm{out}}, b, T(\mathrm{step})$}
\Require $x \in \mathbb{R}^{B\times L\times d_{\mathrm{in}}}$, $W_r \in \mathbb{R}^{d_{\mathrm{in}}\times E}$
\Ensure $y \in \mathbb{R}^{B\times L\times d_{\mathrm{out}}}$, $\mathcal{L}_{\mathrm{LB}}$, $\mathcal{L}_{\mathrm{Z}}$
\State $R \gets x W_r$
\State $\widehat{R} \gets \operatorname{fast\_scaled\_tanh}(R, 10.0)$
\State $\mathcal{L}_{\mathrm{Z}} \gets \operatorname{mean}(\operatorname{logsumexp}(\widehat{R}, -1)^2)$
\State $\widetilde{R} \gets \widehat{R} / T(\mathrm{step})$
\State $I \gets \operatorname{TopKIndices}(\widetilde{R}, k)$
\State $P \gets \operatorname{Softmax}(\operatorname{Gather}(\widetilde{R}, I), -1)$
\State $\mathcal{L}_{\mathrm{LB}} \gets E \sum_{e=1}^{E} \bar{m}_e \bar{p}_e$
\State $G_{\mathrm{exp}} \gets \operatorname{einsum}(\texttt{'er,rst->est'}, U_{\mathrm{exp}}, \mathcal{G})$
\State $X_{\mathrm{shared}} \gets \operatorname{RMSNorm}(x U_{\mathrm{in}})$
\State $X_{\mathrm{core}} \gets \operatorname{FusedLatentMoE}(X_{\mathrm{shared}}, G_{\mathrm{exp}}, I, P)$
\State $y \gets X_{\mathrm{core}} U_{\mathrm{out}} + b$
\State \Return $y, \mathcal{L}_{\mathrm{LB}}, \mathcal{L}_{\mathrm{Z}}$
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
    "./assets/algorithms/appendix_a2_chunk_parallel_scan.png": r"""
\begin{algorithm}[H]
\caption{Chunk-Parallel Scan (SSD)}
\begin{algorithmic}[1]
\Procedure{ChunkParallelScan}{$x, \Delta, A$}
\Require $x \in \mathbb{R}^{B\times L\times H\times P}$, $\Delta \in \mathbb{R}^{B\times L\times H}$, $A \in \mathbb{R}^{H\times N}$
\Ensure $y \in \mathbb{R}^{B\times L\times H\times P}$
\Statex \textbf{Parameters:} chunk size $C = 64$
\State $M \gets \lceil L / C \rceil$; partition $(x, \Delta)$ into chunks $\{\mathcal{C}_m\}_{m=1}^{M}$
\ForAll{$m \in \{1, \dots, M\}$ in parallel}
  \State $\bar{A}^{(m)}_t \gets \exp(\Delta_t A)$ for all $t \in \mathcal{C}_m$
  \State $S^{(m)} \gets \operatorname{AssociativeScan}(\bar{A}^{(m)}, x^{(m)})$
  \State $(\alpha^{(m)}, \beta^{(m)}) \gets \operatorname{ChunkSummary}(S^{(m)})$
\EndFor
\State $s^{(1)}_{\mathrm{in}} \gets 0$
\For{$m \gets 1$ to $M$}
  \If{$m > 1$}
    \State $s^{(m)}_{\mathrm{in}} \gets \alpha^{(m-1)} s^{(m-1)}_{\mathrm{in}} + \beta^{(m-1)}$
  \EndIf
  \State $y^{(m)} \gets \operatorname{ApplyBoundaryState}(S^{(m)}, s^{(m)}_{\mathrm{in}})$
\EndFor
\State \Return $\operatorname{Concat}(y^{(1)}, \dots, y^{(M)})$
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
    "./assets/algorithms/appendix_a3_router_temperature_annealing.png": r"""
\begin{algorithm}[H]
\caption{Router Temperature Annealing}
\begin{algorithmic}[1]
\Procedure{RouterTemperatureAnnealing}{$s, S_{\max}, W, T_{\mathrm{start}}, T_{\mathrm{end}}$}
\Require current step $s$, total steps $S_{\max}$, warmup steps $W$, $T_{\mathrm{start}}$, $T_{\mathrm{end}}$
\Ensure $T(s)$
\If{$s < W$}
  \State \Return $T_{\mathrm{start}}$
\Else
  \State $p \gets \dfrac{s - W}{S_{\max} - W}$
  \State $T(s) \gets T_{\mathrm{end}} + \dfrac{1}{2}(T_{\mathrm{start}} - T_{\mathrm{end}})\bigl(1 + \cos(\pi p)\bigr)$
  \State \Return $T(s)$
\EndIf
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
    "./assets/algorithms/appendix_a4_tuckermoe_backward.png": r"""
\begin{algorithm}[H]
\caption{TuckerMoE Backward Pass}
\begin{algorithmic}[1]
\Procedure{TuckerMoEBackward}{$\delta, X_{\mathrm{shared}}, G_{\mathrm{exp}}, I, P, U_{\mathrm{in}}, U_{\mathrm{exp}}, \mathcal{G}, U_{\mathrm{out}}$}
\Require upstream gradient $\delta = \partial \mathcal{L}/\partial y$, $X_{\mathrm{shared}}$, $G_{\mathrm{exp}}$, selected indices $I$, routing weights $P$
\Statex \hspace{\algorithmicindent} shared factors $U_{\mathrm{in}}, U_{\mathrm{exp}}, \mathcal{G}, U_{\mathrm{out}}$
\Ensure $\partial \mathcal{L}/\partial x$, $\partial \mathcal{L}/\partial U_{\mathrm{in}}$, $\partial \mathcal{L}/\partial U_{\mathrm{exp}}$, $\partial \mathcal{L}/\partial \mathcal{G}$, $\partial \mathcal{L}/\partial U_{\mathrm{out}}$, $\partial \mathcal{L}/\partial P$
\State $\partial \mathcal{L}/\partial X_{\mathrm{shared}} \gets 0$, $\partial \mathcal{L}/\partial G_{\mathrm{exp}} \gets 0$, $\partial \mathcal{L}/\partial U_{\mathrm{out}} \gets 0$
\For{$k \gets 1$ to top-$k$}
  \State $e \gets I[:, k]$
  \State $\Delta_k \gets P[:, k] \odot \delta$
  \State $\partial \mathcal{L}/\partial X_{\mathrm{shared}} \gets \partial \mathcal{L}/\partial X_{\mathrm{shared}} + \Delta_k G_{\mathrm{exp}}[e]^\top$
  \State $\partial \mathcal{L}/\partial P[:, k] \gets \langle \delta, X_{\mathrm{shared}} G_{\mathrm{exp}}[e] U_{\mathrm{out}} \rangle$
  \State $\partial \mathcal{L}/\partial G_{\mathrm{exp}}[e] \gets \partial \mathcal{L}/\partial G_{\mathrm{exp}}[e] + X_{\mathrm{shared}}^\top(\Delta_k U_{\mathrm{out}}^\top)$
  \State $\partial \mathcal{L}/\partial U_{\mathrm{out}} \gets \partial \mathcal{L}/\partial U_{\mathrm{out}} + (X_{\mathrm{shared}} G_{\mathrm{exp}}[e])^\top \Delta_k$
\EndFor
\State $(\partial \mathcal{L}/\partial U_{\mathrm{exp}},\, \partial \mathcal{L}/\partial \mathcal{G}) \gets \operatorname{Mode1Backward}(\partial \mathcal{L}/\partial G_{\mathrm{exp}}, U_{\mathrm{exp}}, \mathcal{G})$
\State $(\partial \mathcal{L}/\partial x,\, \partial \mathcal{L}/\partial U_{\mathrm{in}}) \gets \operatorname{BackpropSharedProjection}(X_{\mathrm{shared}}, \partial \mathcal{L}/\partial X_{\mathrm{shared}})$
\State \Return all gradients
\EndProcedure
\end{algorithmic}
\end{algorithm}
""".strip(),
}

UNICODE_TEXT_MATH_REPLACEMENTS = {
    "→": r"$\to$",
    "←": r"$\leftarrow$",
}


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def sanitize_svg_markup(svg_markup: str) -> str:
    # XML parser used in pandoc/LaTeX SVG path does not accept HTML named entities.
    # Replace common HTML-only entities with XML-safe forms.
    return svg_markup.replace("&nbsp;", "&#160;")


def embed_runtime_svg_styles(svg_markup: str, runtime_css: str) -> str:
    if not runtime_css.strip():
        return svg_markup
    style_block = f"<style>{runtime_css}</style>"
    insert_at = svg_markup.find(">")
    if insert_at == -1:
        return svg_markup
    return svg_markup[: insert_at + 1] + style_block + svg_markup[insert_at + 1 :]


def render_svg_from_html(
    host: str = "127.0.0.1",
    port: int = 8765,
    timeout_ms: int = 45000,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: playwright. Install with `python3 -m pip install playwright` "
            "then run `python3 -m playwright install chromium`."
        ) from exc

    handler = http.server.SimpleHTTPRequestHandler
    old_cwd = os.getcwd()
    os.chdir(PROJECT_DIR)
    server = ReusableTCPServer((host, port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 2400, "height": 1600}, device_scale_factor=2
            )
            page = context.new_page()

            for html_rel_path, svg_rel_path in DIAGRAM_SPECS:
                url = f"http://{host}:{port}/{html_rel_path}"
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                page.wait_for_timeout(1500)
                page.evaluate(
                    """
                    () => {
                      if (window.MathJax && window.MathJax.typesetPromise) {
                        return window.MathJax.typesetPromise();
                      }
                    }
                    """
                )

                svg_markup = page.eval_on_selector("svg", "el => el.outerHTML")
                svg_markup = sanitize_svg_markup(svg_markup)
                mjx_css = page.evaluate(
                    """
                    () => {
                      const el = document.querySelector('style#MJX-SVG-styles');
                      return el ? el.textContent : '';
                    }
                    """
                )
                shared_css = """
                .math-box{
                  display:flex;
                  justify-content:center;
                  align-items:center;
                  width:100%;
                  height:100%;
                  white-space:nowrap;
                }
                """
                svg_markup = embed_runtime_svg_styles(svg_markup, f"{mjx_css}\n{shared_css}")
                output_path = PROJECT_DIR / svg_rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(svg_markup + "\n", encoding="utf-8")
                print(f"Rendered SVG: {output_path.relative_to(PROJECT_DIR)}")

                # Also export high-res PNG fallback.
                png_path = output_path.with_suffix(".png")
                svg_el = page.locator("svg").first
                svg_el.screenshot(path=str(png_path), scale="device")
                print(f"Rendered PNG: {png_path.relative_to(PROJECT_DIR)}")

                # Export vector PDF directly from source HTML page after MathJax/CSS
                # are fully applied, to avoid style loss from stripped SVG documents.
                size = page.evaluate(
                    """
                    () => {
                      const svg = document.querySelector("svg");
                      if (!svg) return { w: 1600, h: 900 };
                      const vb = svg.viewBox && svg.viewBox.baseVal;
                      if (vb && vb.width > 0 && vb.height > 0) {
                        return { w: vb.width, h: vb.height };
                      }
                      const r = svg.getBoundingClientRect();
                      return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
                    }
                    """
                )
                w = float(size["w"])
                h = float(size["h"])
                page.add_style_tag(
                    content=f"""
                    @page {{ margin: 0; }}
                    html, body {{
                        margin: 0 !important;
                        padding: 0 !important;
                        background: #fff !important;
                    }}
                    body {{
                        display: block !important;
                    }}
                    .diagram-wrap {{
                        margin: 0 !important;
                        padding: 0 !important;
                        max-width: none !important;
                    }}
                    svg {{
                        width: {w}px !important;
                        min-width: 0 !important;
                        height: {h}px !important;
                        display: block !important;
                    }}
                    """
                )
                pdf_path = output_path.with_suffix(".pdf")
                page.pdf(
                    path=str(pdf_path),
                    width=f"{w}px",
                    height=f"{h}px",
                    print_background=True,
                    prefer_css_page_size=True,
                    margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                )
                print(f"Rendered vector PDF: {pdf_path.relative_to(PROJECT_DIR)}")

            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        os.chdir(old_cwd)


def build_pdf(
    report_md: Path,
    output_pdf: Path,
    cjk_font: str,
    diagram_format: str,
    main_font: str,
    math_font: str,
) -> None:
    pandoc_cmd = shutil.which("pandoc")
    if not pandoc_cmd:
        raise RuntimeError("pandoc not found. Install pandoc to build PDF.")

    report_text = report_md.read_text(encoding="utf-8")
    for old, new in UNICODE_TEXT_MATH_REPLACEMENTS.items():
        report_text = report_text.replace(old, new)
    report_text = report_text.replace(
        METHOD_PIPELINE_MD, f"```{{=latex}}\n{METHOD_PIPELINE_LATEX}\n```"
    )
    report_text = report_text.replace(
        f"{MODEL_ARCH_MD}\n{MODEL_ARCH_CAPTION_MD}",
        f"```{{=latex}}\n{MODEL_ARCH_LATEX_TEMPLATE}\n```",
    )
    for image_path, latex_snippet in APPENDIX_ALGO_LATEX.items():
        report_text = report_text.replace(
            f"![]({image_path})", f"```{{=latex}}\n{latex_snippet}\n```"
        )
    report_text = report_text.replace(
        "./assets/method_flowchart.svg", f"./assets/method_flowchart.{diagram_format}"
    )
    report_text = report_text.replace(
        "./assets/images/architecture.svg", f"./assets/images/architecture.{diagram_format}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        encoding="utf-8",
        delete=False,
        dir=str(PROJECT_DIR),
    ) as tmp_md:
        tmp_md.write(report_text)
        tmp_md_path = Path(tmp_md.name)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tex",
        encoding="utf-8",
        delete=False,
        dir=str(PROJECT_DIR),
    ) as header_file:
        header_file.write(
            "\\usepackage{fontspec}\n"
            "\\usepackage{unicode-math}\n"
            f"\\setmainfont{{{main_font}}}\n"
            f"\\setmathfont{{{math_font}}}\n"
            "\\usepackage{amsmath,amssymb,mathtools,bm}\n"
            "\\usepackage{float}\n"
            "\\usepackage{algorithm}\n"
            "\\usepackage[noend]{algpseudocode}\n"
            "\\algrenewcommand\\algorithmicrequire{\\textbf{Require:}}\n"
            "\\algrenewcommand\\algorithmicensure{\\textbf{Ensure:}}\n"
            "\\algrenewcommand\\algorithmicreturn{\\textbf{return}}\n"
        )
        header_path = Path(header_file.name)

    cmd = [
        pandoc_cmd,
        str(tmp_md_path),
        "--from",
        "markdown+tex_math_dollars",
        "--pdf-engine=xelatex",
        "--include-in-header",
        str(header_path),
        "-V",
        f"CJKmainfont={cjk_font}",
        "-V",
        "CJKoptions=AutoFakeBold,AutoFakeSlant",
        "-V",
        "geometry:margin=1in",
        "--resource-path=.",
        "--output",
        str(output_pdf),
    ]
    try:
        subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
    finally:
        tmp_md_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
    print(f"Built PDF: {output_pdf.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render high-precision SVG diagrams and export report PDF."
    )
    parser.add_argument(
        "--skip-svg",
        action="store_true",
        help="Skip SVG rendering and only build PDF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PDF,
        help="Output PDF path (default: report.pdf).",
    )
    parser.add_argument(
        "--cjk-font",
        default="Songti SC",
        help="CJK font family used by pandoc/xelatex (default: Songti SC).",
    )
    parser.add_argument(
        "--diagram-format",
        choices=("pdf", "png"),
        default="pdf",
        help="Diagram format used in report PDF build (default: pdf).",
    )
    parser.add_argument(
        "--main-font",
        default="STIX Two Text",
        help="Main Latin font for PDF text rendering (default: STIX Two Text).",
    )
    parser.add_argument(
        "--math-font",
        default="STIX Two Math",
        help="Math font for Unicode/math symbols (default: STIX Two Math).",
    )
    args = parser.parse_args()

    if not args.skip_svg:
        render_svg_from_html()

    build_pdf(
        REPORT_MD,
        args.output.resolve(),
        args.cjk_font,
        args.diagram_format,
        args.main_font,
        args.math_font,
    )


if __name__ == "__main__":
    main()
