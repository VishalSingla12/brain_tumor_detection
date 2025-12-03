import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_full_backward_hook(self.save_gradients)
        self.target_layer.register_forward_hook(self.save_activations)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        return heatmap, output

def generate_gradcam(model, img_tensor, target_class=None):
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)
    heatmap, prediction = grad_cam(img_tensor, target_class)
    img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    orig_img = img_tensor.cpu().numpy().squeeze().transpose(1, 2, 0)
    orig_img = std * orig_img + mean
    orig_img = np.clip(orig_img, 0, 1)
    orig_img = (orig_img * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    return heatmap, Image.fromarray(overlay), prediction
