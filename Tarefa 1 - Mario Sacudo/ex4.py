# Tarefa da aula 6

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def rgb_to_hsv(img):
    altura, largura, _ = img.shape
    hsv = np.zeros_like(img, dtype=float)
    
    # Normalizandor
    r = img[:, :, 0] / 255.0
    g = img[:, :, 1] / 255.0
    b = img[:, :, 2] / 255.0
    
    # Max Min
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    delta = max_rgb - min_rgb
    
    hsv[:, :, 2] = max_rgb
    
    hsv[:, :, 1] = np.where(max_rgb > 0, delta / max_rgb, 0)
    
    # Inicializa H com zeros
    h = np.zeros((altura, largura))
    
    # max == r
    mask = (max_rgb > 0) & (delta > 0) & (max_rgb == r)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6
    
    # max == g
    mask = (max_rgb > 0) & (delta > 0) & (max_rgb == g)
    h[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2
    
    # max == b
    mask = (max_rgb > 0) & (delta > 0) & (max_rgb == b)
    h[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4
    
    h = h / 6.0
    
    hsv[:, :, 0] = h
    
    hsv_uint8 = (hsv * 255).astype(np.uint8)
    
    return hsv_uint8


img = np.array(Image.open('mario_sacudo.png'))[:, :, :3]


hsv_img = rgb_to_hsv(img)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.title('RGB')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('HSV')
plt.imshow(hsv_img)
plt.axis('off')

plt.tight_layout()
plt.show()

Image.fromarray(hsv_img).save('mario_sacudo_hsv.png')
