import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Number of GPUs Available:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("This installation is CPU-only.")