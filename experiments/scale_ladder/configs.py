"""Scale ladder model configurations for Phase 1: behavioral subspace emergence."""

SCALE_CONFIGS = {
    "3M": {
        "n_embd": 64, "n_layer": 2, "n_head": 2,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 50_000_000,
        "batch_size": 32, "lr": 8e-4,
    },
    "4M": {
        "n_embd": 80, "n_layer": 2, "n_head": 2,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 60_000_000,
        "batch_size": 32, "lr": 7.5e-4,
    },
    "4.5M": {
        "n_embd": 88, "n_layer": 2, "n_head": 2,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 65_000_000,
        "batch_size": 32, "lr": 7.25e-4,
    },
    "5M": {
        "n_embd": 96, "n_layer": 2, "n_head": 2,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 75_000_000,
        "batch_size": 32, "lr": 7e-4,
    },
    "7M": {
        "n_embd": 128, "n_layer": 4, "n_head": 4,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 100_000_000,
        "batch_size": 32, "lr": 6e-4,
    },
    "12M": {
        "n_embd": 192, "n_layer": 6, "n_head": 6,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 100_000_000,
        "batch_size": 32, "lr": 6e-4,
    },
    "18M": {
        "n_embd": 256, "n_layer": 6, "n_head": 4,
        "n_positions": 512, "vocab_size": 50257,
        "train_tokens": 200_000_000,
        "batch_size": 32, "lr": 5e-4,
    },
    "34M": {
        "n_embd": 384, "n_layer": 8, "n_head": 6,
        "n_positions": 1024, "vocab_size": 50257,
        "train_tokens": 300_000_000,
        "batch_size": 16, "lr": 3e-4,
    },
    "64M": {
        "n_embd": 512, "n_layer": 12, "n_head": 8,
        "n_positions": 1024, "vocab_size": 50257,
        "train_tokens": 500_000_000,
        "batch_size": 16, "lr": 3e-4,
    },
    "153M": {
        "n_embd": 768, "n_layer": 16, "n_head": 12,
        "n_positions": 1024, "vocab_size": 50257,
        "train_tokens": 1_000_000_000,
        "batch_size": 8, "lr": 2e-4,
    },
    "210M": {
        "n_embd": 768, "n_layer": 24, "n_head": 12,
        "n_positions": 1024, "vocab_size": 50257,
        "train_tokens": 1_000_000_000,
        "batch_size": 8, "lr": 2e-4,
    },
}

# Ordered by size for sequential training
SCALE_ORDER = ["3M", "4M", "4.5M", "5M", "7M", "12M", "18M", "34M", "64M", "153M", "210M"]

# Behaviors that support contrast-pair subspace extraction on base models
SUBSPACE_BEHAVIORS = ["factual", "sycophancy", "toxicity", "bias"]
