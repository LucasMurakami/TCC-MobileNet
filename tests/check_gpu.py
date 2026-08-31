import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("Apple Silicon (MPS) Available: True")
else:
    print("Running on CPU")

