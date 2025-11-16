import torch
print(torch.version.cuda)        # Shows CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))