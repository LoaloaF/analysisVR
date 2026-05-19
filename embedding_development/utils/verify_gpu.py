import sys

# ── PyTorch / CUDA ──────────────────────────────────────────────────────────
try:
    import torch
    print(f"PyTorch version       : {torch.__version__}")
    print(f"CUDA available        : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — GPU not available.")
        print("Do not run the training scripts on a CPU-only setup.")
        sys.exit(1)
    print(f"CUDA device count     : {torch.cuda.device_count()}")
    print(f"Device name           : {torch.cuda.get_device_name(0)}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total GPU memory      : {total_mem_gb:.2f} GB")
except Exception as e:
    print(f"ERROR during PyTorch/CUDA check: {e}")
    sys.exit(1)

# ── CEBRA ───────────────────────────────────────────────────────────────────
try:
    import cebra
    import cebra.models
    print(f"\nCEBRA version         : {cebra.__version__}")
except Exception as e:
    print(f"ERROR importing CEBRA: {e}")
    sys.exit(1)

try:
    encoder = cebra.models.init('offset10-model', num_neurons=10, num_units=32, num_output=8)
    print("CEBRA encoder init    : OK")
except Exception as e:
    print(f"ERROR initialising CEBRA encoder: {e}")
    sys.exit(1)

# ── CEBRA encoder on GPU ────────────────────────────────────────────────────
try:
    encoder = encoder.to('cuda')
    dummy = torch.randn(4, 10, 10, device='cuda')
    with torch.no_grad():
        out = encoder(dummy)
    if out.dim() == 3:
        out = out.squeeze(-1)
    print(f"CEBRA forward pass    : OK — output shape {tuple(out.shape)}, device {out.device}")
except Exception as e:
    print(f"ERROR during CEBRA GPU forward pass: {e}")
    sys.exit(1)

print("\nAll checks passed — GPU pipeline is ready.")
