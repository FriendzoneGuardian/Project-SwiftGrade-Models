import importlib
import warnings
from colorama import Fore, Style, init

# Colorama setup
init(autoreset=True)

print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
print(Fore.MAGENTA + Style.BRIGHT + "GPU AVAILABILITY CHECK".center(60))
print(Fore.CYAN + Style.BRIGHT + "=" * 60)

has_gpu = False
can_use_gpu = False
gpu_details = []

# Check PyTorch GPU
try:
    import torch
    if torch.cuda.is_available():
        has_gpu = True
        can_use_gpu = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_details.append(f"PyTorch CUDA Enabled: {gpu_name}")
    else:
        gpu_details.append("PyTorch found but CUDA not available.")
except ImportError:
    gpu_details.append("PyTorch not installed.")

# Check TensorFlow GPU
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        has_gpu = True
        can_use_gpu = True
        for gpu in gpus:
            gpu_details.append(f"TensorFlow GPU Device: {gpu.name}")
    else:
        gpu_details.append("TensorFlow found but no GPU available.")
except ImportError:
    gpu_details.append("TensorFlow not installed.")

# Optional: Add GPUtil for more hardware info
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        has_gpu = True
        for gpu in gpus:
            gpu_details.append(f"Detected GPU: {gpu.name} ({gpu.memoryTotal}MB, Load: {gpu.load*100:.1f}%)")
except ImportError:
    gpu_details.append("GPUtil not installed (optional).")

# Result Summary
print(Fore.LIGHTYELLOW_EX + "\nSystem GPU Summary:")
print(Fore.CYAN + "-" * 60)
for detail in gpu_details:
    print(Fore.WHITE + "  • " + detail)

# Boolean Results
print(Fore.LIGHTGREEN_EX if has_gpu else Fore.LIGHTRED_EX)
print(f"\nGPU Detected? {'Yes' if has_gpu else 'No'}")
print(f"Usable for ML? {'Yes' if can_use_gpu else 'No'}")

print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
