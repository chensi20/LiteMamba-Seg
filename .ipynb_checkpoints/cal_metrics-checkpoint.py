import os
import time
import torch
import numpy as np
from thop import profile

from config import Config
from models.D4Placement import D4Placement_UNet
from models.MambaSeg import MambaSeg_UNet


# ============================================================
# Model builder
# ============================================================
def get_model(name):

    controlled_positions = [
        "none",
        "pre_up",
        "pre_fusion",
        "post_fusion",
        "post_conv",
    ]

    # --------------------------------------------------------
    # Controlled D4 placement models
    # --------------------------------------------------------
    if name in controlled_positions:

        model = D4Placement_UNet(
            n_classes=Config.NUM_CLASSES,
            d4_mamba_pos=name,
            use_layer3_mamba=False,
            use_bottleneck_mamba=True,
            use_conv_refine=True,
            pretrained=True,
        )

        return model

    # --------------------------------------------------------
    # Original MambaSeg models
    # --------------------------------------------------------
    if name == "2025":
        # Final model
        model = MambaSeg_UNet(
            n_classes=Config.NUM_CLASSES,
            use_layer3_mamba=False,
            use_d4_mamba=True,
            use_bottleneck_mamba=True,
            use_conv_refine=True,
        )

    elif name == "Conv":
        # Conv refinement only
        model = MambaSeg_UNet(
            n_classes=Config.NUM_CLASSES,
            use_layer3_mamba=False,
            use_d4_mamba=False,
            use_bottleneck_mamba=False,
            use_conv_refine=True,
        )

    elif name == "Layer3":
        # Layer3 Mamba
        model = MambaSeg_UNet(
            n_classes=Config.NUM_CLASSES,
            use_layer3_mamba=True,
            use_d4_mamba=False,
            use_bottleneck_mamba=True,
            use_conv_refine=True,
        )

    elif name == "Conv_Bottleneck":
        # Conv refinement + bottleneck Mamba
        model = MambaSeg_UNet(
            n_classes=Config.NUM_CLASSES,
            use_layer3_mamba=False,
            use_d4_mamba=False,
            use_bottleneck_mamba=True,
            use_conv_refine=True,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model


# ============================================================
# Load checkpoint
# ============================================================
def load_checkpoint(model, name, device):

    weights_path = f"./checkpoints/best_model_{name}.pth"

    if not os.path.exists(weights_path):
        print(
            f"Notice: Checkpoint {weights_path} not found. "
            f"Testing architecture only."
        )
        return model

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    print(f"Loaded weights: {weights_path}")

    return model


# ============================================================
# Measure one model
# ============================================================
def measure_one(name):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    input_tensor = torch.randn(
        1,
        3,
        352,
        352,
        device=device,
    )

    print("\n" + "=" * 65)
    print(f"Testing Model: {name}")
    print("=" * 65)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = get_model(name).to(device)

    # --------------------------------------------------------
    # Load corresponding weights
    # --------------------------------------------------------
    model = load_checkpoint(
        model,
        name,
        device,
    )

    model.eval()

    # --------------------------------------------------------
    # Params
    # --------------------------------------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------
    try:
        flops, thop_params = profile(
            model,
            inputs=(input_tensor,),
            verbose=False,
        )

        gflops = flops / 1e9
        thop_params_m = thop_params / 1e6

    except Exception as e:
        print(f"THOP failed: {e}")

        gflops = float("nan")
        thop_params_m = float("nan")

    print(f"Manual Params    : {total_params / 1e6:.4f} M")
    print(f"Trainable Params : {trainable_params / 1e6:.4f} M")
    print(f"THOP Params      : {thop_params_m:.4f} M")
    print(f"THOP GFLOPs      : {gflops:.4f} G")

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    warmup = 100

    print("-" * 65)
    print(f"GPU warming up ({warmup} iterations)...")

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------
    iterations = 300
    latencies_ms = []

    print(
        f"Starting inference loop for "
        f"{iterations} iterations..."
    )

    with torch.no_grad():

        if torch.cuda.is_available():

            starter = torch.cuda.Event(
                enable_timing=True
            )

            ender = torch.cuda.Event(
                enable_timing=True
            )

            for _ in range(iterations):

                starter.record()

                _ = model(input_tensor)

                ender.record()

                torch.cuda.synchronize()

                latencies_ms.append(
                    starter.elapsed_time(ender)
                )

        else:

            for _ in range(iterations):

                start = time.perf_counter()

                _ = model(input_tensor)

                end = time.perf_counter()

                latencies_ms.append(
                    (end - start) * 1000.0
                )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    avg_latency = float(
        np.mean(latencies_ms)
    )

    std_latency = float(
        np.std(latencies_ms)
    )

    median_latency = float(
        np.median(latencies_ms)
    )

    fps = (
        1000.0 / avg_latency
        if avg_latency > 0
        else float("nan")
    )

    print("=" * 65)
    print(f"Model               : {name}")
    print(f"Params              : {trainable_params / 1e6:.4f} M")
    print(f"GFLOPs              : {gflops:.4f} G")
    print(
        f"Latency (mean)      : "
        f"{avg_latency:.3f} ± {std_latency:.3f} ms"
    )
    print(
        f"Latency (median)    : "
        f"{median_latency:.3f} ms"
    )
    print(f"FPS                 : {fps:.2f}")
    print("=" * 65)

    result = {
        "name": name,
        "params": trainable_params / 1e6,
        "gflops": gflops,
        "latency": avg_latency,
        "std": std_latency,
        "median": median_latency,
        "fps": fps,
    }

    del model
    del input_tensor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Measure all models
# ============================================================
def measure_all():

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print(f"Device: {device_name}")

    model_names = [
        "none",
        "pre_up",
        "pre_fusion",
        "post_fusion",
        "post_conv",
        "2025",
        "Conv",
        "Layer3",
        "Conv_Bottleneck",
    ]

    results = []

    for name in model_names:

        try:
            result = measure_one(name)

            if result is not None:
                results.append(result)

        except Exception as e:
            print(f"\nFailed on {name}: {e}")

    print("\n")
    print("=" * 90)
    print("FINAL PROFILING SUMMARY")
    print("=" * 90)

    print(
        f"{'Model':<20}"
        f"{'Params(M)':>12}"
        f"{'GFLOPs':>12}"
        f"{'Latency(ms)':>16}"
        f"{'FPS':>12}"
    )

    print("-" * 90)

    for r in results:
        print(
            f"{r['name']:<20}"
            f"{r['params']:>12.4f}"
            f"{r['gflops']:>12.4f}"
            f"{r['latency']:>16.3f}"
            f"{r['fps']:>12.2f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    measure_all()