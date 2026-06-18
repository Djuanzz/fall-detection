"""Ukur kompleksitas model BlockGCN_Base: params, FLOPs, VRAM, latensi/FPS.
Untuk bab 4.1.4 & 4.1.5 buku TA. Jalankan di env block-gcn.
"""
import sys, time, importlib
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def import_class(name):
    parts = name.split('.')
    mod = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


class _Tee:
    """Tulis output ke stdout sekaligus ke file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    out_path = ROOT / "docs/eval/komputasi_04.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(out_path, "w", encoding="utf-8")
    _stdout = sys.stdout
    sys.stdout = _Tee(_stdout, out_file)

    exp_dir = ROOT / "weights_new/04_17ful"
    cfg_path = exp_dir / "config.yaml"
    weights_path = exp_dir / "runs-38-4142.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"])

    # Load bobot checkpoint 04 (metrik params/FLOPs/VRAM/latensi tetap sama,
    # tapi ini memastikan yang diukur benar2 model final 04).
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Weights          : {weights_path.name}")

    # --- Parameter count ---
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model            : {cfg['model']}")
    print(f"window_size      : {cfg['model_args']['window_size']}")
    print(f"num_point        : {cfg['model_args']['num_point']}")
    print(f"Total params     : {total:,} ({total/1e6:.3f} M)")
    print(f"Trainable params : {trainable:,} ({trainable/1e6:.3f} M)")
    print(f"Model size (fp32): {total*4/1e6:.2f} MB")

    # --- Input dummy (N, C, T, V, M) ---
    T = cfg["model_args"]["window_size"]
    V = cfg["model_args"]["num_point"]
    x = torch.randn(1, 3, T, V, 1)

    # --- FLOPs (coba thop, fallback ke ptflops, fallback skip) ---
    flops = None
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        flops = macs * 2
        print(f"MACs             : {macs/1e9:.3f} G")
        print(f"FLOPs (2*MACs)   : {flops/1e9:.3f} GFLOPs")
    except Exception as e:
        print(f"[thop gagal: {e}]")

    # --- Latensi & FPS (CPU + CUDA jika ada) ---
    for dev in (["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])):
        m = model.to(dev)
        xd = x.to(dev)
        with torch.no_grad():
            for _ in range(5):           # warmup
                m(xd)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            N = 50
            for _ in range(N):
                m(xd)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            dt = (time.time() - t0) / N
        print(f"[{dev}] latensi/inference: {dt*1000:.2f} ms | throughput: {1/dt:.1f} inf/s")
        if dev.startswith("cuda"):
            mem = torch.cuda.max_memory_allocated(dev) / 1e6
            print(f"[{dev}] peak VRAM        : {mem:.1f} MB")

    print(f"\nDevice CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sys.stdout = _stdout
    out_file.close()
    print(f"\n[Hasil tersimpan ke: {out_path}]")


if __name__ == "__main__":
    main()
