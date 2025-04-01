# Tarefa da aula 5 - Ex1

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def mirror(img, reverse_x=False, reverse_y=False):
    mirrored_img = img.copy()
    
    if reverse_x:
        mirrored_img = mirrored_img[:, ::-1, :]
    
    if reverse_y:
        mirrored_img = mirrored_img[::-1, :, :]
    
    return mirrored_img

img = np.array(Image.open('mario_sacudo.png'))[:, :, :3]

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
plt.title('Original')
plt.imshow(img)


img_x = mirror(img, reverse_x=True)
plt.subplot(2, 2, 2)
plt.title('Mirror X')
plt.imshow(img_x)


img_y = mirror(img, reverse_y=True)
plt.subplot(2, 2, 3)
plt.title('Mirror Y')
plt.imshow(img_y)


img_xy = mirror(img, reverse_x=True, reverse_y=True)
plt.subplot(2, 2, 4)
plt.title('Mirror X and Y')
plt.imshow(img_xy)


plt.tight_layout()
plt.show()


Image.fromarray(img_x).save('mirror_x.png')
Image.fromarray(img_y).save('mirror_y.png')
Image.fromarray(img_xy).save('mirror_xy.png')