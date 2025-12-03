import torch
import random
import numpy as np
import os
import cv2
from PIL import Image
import config

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def seed_everything(seed=config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_image(image_path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize(target_size)
    return image

def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)
