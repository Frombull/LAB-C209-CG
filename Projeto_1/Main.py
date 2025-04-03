import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


IMAGES = [
    "Projeto_1/P1.png",
    "Projeto_1/P2.png",
    "Projeto_1/P3.jpg",
    "Projeto_1/P4.png"
]


def load_images() -> tuple:
    """Carrega os 4 esquisitos e converte para RGB, se precisar"""

    images = []
    for path in IMAGES:
        img = Image.open(path)

        if img.mode != "RGB":
            img = img.convert("RGB")
            
        images.append(np.array(img))
    
    return tuple(images)


def convert_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    # Carregar a imagem
    l, c, _ = rgb_image.shape
    
    # Criar matriz para armazenar a versão em cinza
    grayscale_image = np.zeros((l, c), dtype=np.uint8)
    
    # Converter para escala de cinza
    for i in range(l):
        for j in range(c):
            r, g, b = map(float, rgb_image[i, j])
            grayscale_image[i, j] = int((r + g + b) / 3)
    
    return grayscale_image


if __name__ == "__main__":
    IMG_1, IMG_2, IMG_3, IMG_4 = load_images()

    image_grayscale = convert_to_grayscale(IMG_2)
    
    plt.figure(figsize=(16, 16))
    plt.imshow(image_grayscale, cmap='gray')
    # plt.axis('off')
    plt.show()
