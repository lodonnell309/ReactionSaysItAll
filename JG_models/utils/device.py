import torch

def get_device():
    # Prioritize MPS (Apple), then CUDA (NVIDIA), then CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)}).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device
