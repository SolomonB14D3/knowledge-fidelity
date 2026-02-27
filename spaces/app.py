"""rho-eval — HF Spaces Gradio Demo.

Compress an LLM and audit whether it still knows truth vs popular myths.
"""

import json
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]

PROBE_SETS = ["default", "mandela", "medical", "all"]


def get_probes(probe_set: str):
    from rho_eval.probes import (
        get_default_probes, get_mandela_probes, get_medical_probes,
        get_all_probes,
    )
    probe_map = {
        "default": get_default_probes,
        "mandela": get_mandela_probes,
        "medical": get_medical_probes,
        "all": get_all_probes,
    }
    # Try commonsense if available
    try:
        from rho_eval.probes import get_commonsense_probes
        probe_map["commonsense"] = get_commonsense_probes
    except ImportError:
        pass

    if probe_set not in probe_map:
        return get_default_probes()
    return probe_map[probe_set]()


def make_bar_chart(report, probes):
    """Create before/after confidence comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    before = report["audit_before"]
    after = report["audit_after"]
    x = np.arange(len(probes))
    labels = [p.get("id", f"probe_{i}")[:15] for i, p in enumerate(probes)]

    # Before
    ax = axes[0]
    ax.bar(x - 0.2, before["true_confs"], 0.4, label="True", color="#2ecc71", alpha=0.8)
    ax.bar(x + 0.2, before["false_confs"], 0.4, label="False", color="#e74c3c", alpha=0.8)
    ax.set_title(f'BEFORE (rho={report["rho_before"]:.3f})', fontsize=12)
    ax.set_ylabel("Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()

    # After
    ax = axes[1]
    ax.bar(x - 0.2, after["true_confs"], 0.4, label="True", color="#2ecc71", alpha=0.8)
    ax.bar(x + 0.2, after["false_confs"], 0.4, label="False", color="#e74c3c", alpha=0.8)
    ax.set_title(f'AFTER CF90 (rho={report["rho_after"]:.3f})', fontsize=12)
    ax.set_ylabel("Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()

    plt.suptitle("rho-eval: Confidence Before vs After Compression", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def run_compress_and_audit(model_name, ratio, probe_set, progress=gr.Progress()):
    """Main demo function."""
    progress(0.05, desc="Loading probes...")
    probes = get_probes(probe_set)

    progress(0.1, desc=f"Loading {model_name}...")
    from rho_eval import compress_and_audit

    report = compress_and_audit(
        model_name,
        ratio=ratio,
        probes=probes,
        device="cpu",
    )

    progress(0.9, desc="Generating report...")

    # Summary text
    rho_change = report["rho_after"] - report["rho_before"]
    direction = "IMPROVED" if rho_change > 0 else "dropped"
    summary = (
        f"Model: {model_name}\n"
        f"Compression: {ratio:.0%} rank | {report['compression']['n_compressed']} matrices\n"
        f"Layers frozen: {report['freeze']['n_frozen']}/{report['freeze']['n_layers']}\n"
        f"Retention: {report['retention']:.0%}\n"
        f"rho: {report['rho_before']:.3f} → {report['rho_after']:.3f} "
        f"({direction} by {abs(rho_change):.3f})\n"
        f"Time: {report['elapsed_seconds']:.1f}s"
    )

    if rho_change > 0:
        summary += f"\n\n*** DENOISING DETECTED: rho {direction} by {rho_change:.3f} ***"

    # Bar chart
    fig = make_bar_chart(report, probes)

    # Per-probe table
    rows = []
    for i, p in enumerate(probes):
        db = report["audit_before"]["deltas"][i]
        da = report["audit_after"]["deltas"][i]
        status = "OK" if da > 0 else ("FLIP" if db > 0 else "was neg")
        rows.append({
            "Probe": p.get("id", f"probe_{i}"),
            "Delta Before": f"{db:+.4f}",
            "Delta After": f"{da:+.4f}",
            "Status": status,
        })
    df = pd.DataFrame(rows)

    progress(1.0, desc="Done!")
    return summary, fig, df


def run_audit_only(model_name, probe_set, progress=gr.Progress()):
    """Audit without compression (baseline)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rho_eval import audit_model

    progress(0.1, desc=f"Loading {model_name}...")
    probes = get_probes(probe_set)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()

    progress(0.5, desc="Running audit...")
    audit = audit_model(model, tokenizer, probes=probes)

    summary = (
        f"Model: {model_name} (no compression)\n"
        f"Probes: {probe_set} ({len(probes)})\n"
        f"rho: {audit['rho']:.3f} (p={audit['rho_p']:.4f})\n"
        f"Mean delta: {audit['mean_delta']:.4f}\n"
        f"Positive: {audit['n_positive_delta']}/{audit['n_probes']}"
    )

    rows = []
    for i, p in enumerate(probes):
        d = audit["deltas"][i]
        rows.append({
            "Probe": p.get("id", f"probe_{i}"),
            "Delta": f"{d:+.4f}",
            "Correct": "Yes" if d > 0 else "No",
        })
    df = pd.DataFrame(rows)

    progress(1.0, desc="Done!")
    return summary, None, df


# --- Gradio UI ---

with gr.Blocks(title="rho-eval", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# rho-eval\n"
        "**Compress an LLM while auditing whether it still knows truth vs popular myths.**\n\n"
        "Uses the same factual probes for SVD compression importance scoring and "
        "behavioral false-belief detection."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=MODELS, value=MODELS[0], label="Model",
                allow_custom_value=True,
            )
            ratio_slider = gr.Slider(
                minimum=0.5, maximum=0.95, value=0.7, step=0.05,
                label="Compression Ratio (0.7 = keep 70% of singular values)",
            )
            probe_dropdown = gr.Dropdown(
                choices=PROBE_SETS, value="default", label="Probe Set",
            )
            with gr.Row():
                compress_btn = gr.Button("Compress + Audit", variant="primary")
                audit_btn = gr.Button("Audit Only (baseline)")

    with gr.Row():
        summary_box = gr.Textbox(label="Summary", lines=8, interactive=False)

    with gr.Row():
        chart = gr.Plot(label="Before vs After Confidence")

    with gr.Row():
        results_table = gr.Dataframe(label="Per-Probe Results")

    compress_btn.click(
        fn=run_compress_and_audit,
        inputs=[model_choice, ratio_slider, probe_dropdown],
        outputs=[summary_box, chart, results_table],
    )
    audit_btn.click(
        fn=run_audit_only,
        inputs=[model_choice, probe_dropdown],
        outputs=[summary_box, chart, results_table],
    )


if __name__ == "__main__":
    demo.launch()
