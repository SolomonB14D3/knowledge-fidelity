# MLX Inference Recipe (Apple Silicon)

Run your CF90-compressed model on Apple Silicon using MLX for fast inference.

## Prerequisites

```bash
pip install mlx mlx-lm
```

## Convert & Run

```bash
# Convert compressed HF model to MLX format
python -m mlx_lm.convert --hf-path ./compressed_model --mlx-path ./compressed_model_mlx

# Optional: quantize for even faster inference
python -m mlx_lm.convert --hf-path ./compressed_model --mlx-path ./compressed_model_mlx_q4 -q --q-bits 4

# Run inference
python -m mlx_lm.generate --model ./compressed_model_mlx --prompt "The capital of France is"
```

## Benchmark

```bash
# Compare speeds
python -m mlx_lm.generate --model ./compressed_model_mlx --prompt "Explain quantum computing:" --max-tokens 200 --verbose
```

## Notes

- MLX uses unified memory on Apple Silicon (M1/M2/M3/M4), so larger models fit than on discrete GPUs with equivalent VRAM
- CF90 compression reduces memory footprint and can improve generation quality at 7B+ scale
- Use `--q-bits 4` for best speed/quality tradeoff (CF90 + Q4 validated: 77% fact retention on Llama 7B)
- MPS (PyTorch Metal) has known matmul bugs with some architectures — use MLX for inference, CPU for compression
