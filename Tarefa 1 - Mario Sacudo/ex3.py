# Tarefa da aula 5 - Ex3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def bilinear_scale(img, scale_x, scale_y):
    height, width, channels = img.shape
    new_height, new_width = int(height * scale_y), int(width * scale_x)
    
    scaled_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            src_x = j / scale_x
            src_y = i / scale_y
            
            x1, y1 = int(src_x), int(src_y)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
            
            dx = src_x - x1
            dy = src_y - y1
            
            for c in range(channels):
                val = (1 - dx) * (1 - dy) * img[y1, x1, c] + \
                      dx * (1 - dy) * img[y1, x2, c] + \
                      (1 - dx) * dy * img[y2, x1, c] + \
                      dx * dy * img[y2, x2, c]
                
                scaled_img[i, j, c] = int(val)
    
    return scaled_img

img = np.array(Image.open('mario_sacudo.png'))[:, :, :3]


scale_x, scale_y = 0.1, 0.1
scaled_img = bilinear_scale(img, scale_x, scale_y)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img)


plt.subplot(1, 2, 2)
plt.title(f'Scaled (x{scale_x}, y{scale_y})')
plt.imshow(scaled_img)


plt.tight_layout()
plt.show()


Image.fromarray(scaled_img).save('bilinear_and_scaled.png')