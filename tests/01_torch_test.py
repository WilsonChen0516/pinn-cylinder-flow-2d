import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

# 快速 smoke test
x = torch.randn(1000, 1000, device="cuda")
y = x @ x.T
print(f"GPU matmul OK, result shape: {y.shape}")