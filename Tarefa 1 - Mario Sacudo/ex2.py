# Tarefa da aula 5 - Ex2

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = np.array(Image.open('mario_sacudo.png'))[:, :, :3]
height, width, channels = img.shape

shear_x, shear_y = 0.5, 0.3

new_width = int(width + height * shear_x)
new_height = int(height + width * shear_y)

sheared_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        new_j = int(j + i * shear_x)
        new_i = int(i + j * shear_y)
        if 0 <= new_i < new_height and 0 <= new_j < new_width:
            sheared_img[new_i, new_j] = img[i, j]


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img)


plt.subplot(1, 2, 2)
plt.title(f'Sheared (x={shear_x}, y={shear_y})')
plt.imshow(sheared_img)


plt.tight_layout()
plt.show()


Image.fromarray(sheared_img).save('sheared_image.png')