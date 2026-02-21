#!/usr/bin/env python3
"""
Export a CF90-compressed model to GGUF format for llama.cpp / Ollama.

Requires: pip install llama-cpp-python  (or use convert.py from llama.cpp)

Usage:
    python deployment/export_gguf.py --input compressed_model/ --output model.gguf
    python deployment/export_gguf.py --input compressed_model/ --output model.gguf --quantize q4_k_m
"""

import sys
import argparse
import subprocess
from pathlib import Path


def export_gguf(input_dir: str, output_path: str, quantize: str = None):
    """Export HuggingFace model to GGUF.

    This is a thin wrapper around llama.cpp's convert tools. For production
    use, clone llama.cpp and use convert_hf_to_gguf.py directly.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        sys.exit(1)

    # Try using llama.cpp convert script
    convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print("llama.cpp convert script not found.")
        print("To export GGUF, either:")
        print("  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        print("  2. Run: python llama.cpp/convert_hf_to_gguf.py \\")
        print(f"       {input_dir} --outfile {output_path}")
        if quantize:
            print(f"  3. Quantize: ./llama.cpp/build/bin/llama-quantize {output_path} {output_path.stem}_{quantize}.gguf {quantize}")
        return

    # Convert
    cmd = [
        sys.executable, str(convert_script),
        str(input_dir),
        "--outfile", str(output_path),
    ]
    print(f"Converting: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved to {output_path}")

    # Quantize if requested
    if quantize:
        quantize_bin = Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"
        if quantize_bin.exists():
            q_output = output_path.parent / f"{output_path.stem}_{quantize}.gguf"
            cmd_q = [str(quantize_bin), str(output_path), str(q_output), quantize]
            print(f"Quantizing: {' '.join(cmd_q)}")
            subprocess.run(cmd_q, check=True)
            print(f"Quantized to {q_output}")
        else:
            print(f"Quantize binary not found at {quantize_bin}")
            print(f"Build llama.cpp first, then run:")
            print(f"  llama-quantize {output_path} {output_path.stem}_{quantize}.gguf {quantize}")


def main():
    parser = argparse.ArgumentParser(description="Export CF90 model to GGUF")
    parser.add_argument("--input", required=True, help="Path to HF model directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--quantize", default=None,
                       choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0"],
                       help="Quantization type (optional)")
    args = parser.parse_args()
    export_gguf(args.input, args.output, args.quantize)


if __name__ == "__main__":
    main()
