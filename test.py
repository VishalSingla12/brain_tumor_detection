import os
import torch
import config
from model import BrainTumorCNN
def test():
    print("Running  Test...")
    try:
        import cv2
        import streamlit
        import kagglehub
        print(" Imports successful")
    except ImportError as e:
        print(f" Import failed: {e}")
        return
    try:
        model = BrainTumorCNN(num_classes=4)
        print(" Model initialized successfully")
    except Exception as e:
        print(f" Model initialization failed: {e}")
        return
    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f" Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f" Forward pass failed: {e}")
        return
    required_dirs = ['data', 'logs', 'outputs']
    if os.path.exists('data_loader.py'):
         print(" data_loader.py exists")
    print("\n Test Complete. Ready to run `python data_loader.py` and then `python train.py`.")
if __name__ == "__main__":
    test()
