import torch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime version: {torch.version.cuda}")
    print(f"Selected device: {device}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Quick tensor op to verify computation runs on selected device.
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    z = torch.matmul(x, y)
    print(f"Tensor device: {z.device}")
    print(f"Tensor mean: {z.mean().item():.6f}")


if __name__ == "__main__":
    main()
