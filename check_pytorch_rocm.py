import torch

def check_gpu():
    if torch.cuda.is_available():
        print("PyTorch ROCm support is available.")
        print(f"CUDA (ROCm) version: {torch.version.cuda}") # For ROCm, this shows the CUDA version PyTorch was built against, not ROCm version directly
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current GPU index: {current_device}")
        print(f"Current GPU name: {torch.cuda.get_device_name(current_device)}")
        # Create a tensor and move it to GPU to confirm
        try:
            x = torch.tensor([1.0, 2.0]).to("cuda")
            print(f"Successfully moved a tensor to cuda: {x}")
            print("ROCm is likely working with PyTorch!")
        except Exception as e:
            print(f"Error moving tensor to cuda: {e}")
            print("There might be an issue with the ROCm setup or PyTorch-ROCm compatibility.")
    else:
        print("PyTorch ROCm support is NOT available. PyTorch is using CPU.")
        print("Please ensure you have installed the PyTorch version built for ROCm.")
        print("You can find installation instructions at https://pytorch.org/get-started/locally/")
        print("Make sure to select the correct ROCm version for your setup.")

if __name__ == "__main__":
    check_gpu()
