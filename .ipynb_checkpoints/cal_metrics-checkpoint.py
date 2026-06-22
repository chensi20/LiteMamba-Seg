import torch
import numpy as np
from thop import profile
from config import Config
from models.MambaSeg import MambaSeg_UNet as MyMambaModel


def measure_all():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    input_size = (1, 3, 352, 352)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model = MyMambaModel(
        n_classes=Config.NUM_CLASSES,
        use_layer3_mamba=Config.USE_LAYER3_MAMBA,
        use_d4_mamba=Config.USE_D4_MAMBA,
        use_bottleneck_mamba=Config.USE_BOTTLENECK_MAMBA,
        use_conv_refine=Config.USE_CONV_REFINE,
    ).to(device)
    model.eval()

    input_tensor = torch.randn(*input_size, device=device)

    print("=" * 40)
    print("Calculating Params and GFLOPs ...")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Manual Params    : {total_params / 1e6:.2f} M")
    print(f"Trainable Params : {trainable_params / 1e6:.2f} M")

    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        print(f"THOP Params      : {params / 1e6:.2f} M")
        print(f"GFLOPs           : {flops / 1e9:.2f} G")
    except Exception as e:
        print(f"Failed to calculate FLOPs: {e}")

    print("-" * 40)
    print("GPU warming up (100 iterations)...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    iterations = 300
    print(f"Starting inference loop for {iterations} iterations...")

    latencies_ms = []

    with torch.no_grad():
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            for _ in range(iterations):
                starter.record()
                _ = model(input_tensor)
                ender.record()
                torch.cuda.synchronize()
                latencies_ms.append(starter.elapsed_time(ender))
        else:
            import time
            for _ in range(iterations):
                start = time.time()
                _ = model(input_tensor)
                end = time.time()
                latencies_ms.append((end - start) * 1000.0)

    avg_latency = np.mean(latencies_ms)
    std_latency = np.std(latencies_ms)
    median_latency = np.median(latencies_ms)
    fps = 1000.0 / avg_latency

    print("=" * 40)
    print(f"Latency (mean):   {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"Latency (median): {median_latency:.2f} ms")
    print(f"FPS:              {fps:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    measure_all()