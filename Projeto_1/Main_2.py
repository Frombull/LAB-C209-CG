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
    """Converte de RGB para escala de cinza"""

    l, c, _ = rgb_image.shape
    
    grayscale_image = np.zeros((l, c), dtype=np.uint8)
    
    for i in range(l):
        for j in range(c):
            r, g, b = map(float, rgb_image[i, j])
            grayscale_image[i, j] = int((r + g + b) / 3)
    
    return grayscale_image


def show_image(image, title: str = "Pen gu im") -> None:
    """Mostra a imagem com matplotlib"""

    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.imshow(image)
    plt.show()


def concatenate_images(images):
    """Junta todas as imagens em uma unica"""

    # Usando o tamanho da primeira imagem
    target_height, target_width = 300, 300
    resized_images = []
    
    for img in images:
        height, width = img.shape[0], img.shape[1]
        aspect = width / height
        
        if height > width:
            new_height = target_height
            new_width = int(target_height * aspect)
        else:
            new_width = target_width
            new_height = int(target_width / aspect)
            
        # Criar imagem PIL, redimensionar e converter de volta para numpy
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized_images.append(np.array(pil_img))
    
    # Criar uma imagem em branco para conter todas as imagens com 2 imagens por linha
    rows = (len(images) + 1) // 2
    full_img = np.ones((rows * target_height, 2 * target_width, 3), dtype=np.uint8) * 255
    
    # Adicionar cada imagem redimensionada
    for i, img in enumerate(resized_images):
        row = i // 2
        col = i % 2
        h, w = img.shape[0], img.shape[1]
        
        # Centralizar a imagem em seu espaço
        y_offset = row * target_height + (target_height - h) // 2
        x_offset = col * target_width + (target_width - w) // 2
        
        full_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    return full_img


def change_colors(image):
    """Troca uma cor"""
    
    # Fazendo uma cópia para não modificar o original
    modified = image.copy()
    
    # Azul --> rosa
    dark_blue_mask = (np.abs(image[:,:,0] - 0) < 10) & (np.abs(image[:,:,1] - 50) < 10) & (np.abs(image[:,:,2] - 101) < 10)
    modified[dark_blue_mask] = [255, 105, 180]  # Rosa
    
    return modified


def flip_horizontal(image):
    """Espelhamento horizontal"""
    
    return image[:, ::-1]


def crop_penguin(image):
    """Recorta uma unidade penguina"""
    
    height, width = image.shape[0], image.shape[1]
    
    # Grade 2x2
    half_width = width // 2
    half_height = height // 2
    
    # Recortando
    left = half_width
    top = 0
    right = width
    bottom = half_height
    
    return image[top:bottom, left:right]


def apply_color_threshold(image, threshold):
    """Aplica uma cor nos pixels abaixo do threshold"""

    result = image.copy()
    
    # Calcular a luminância para cada pixel
    r_channel = image[:,:,0].astype(float)
    g_channel = image[:,:,1].astype(float)
    b_channel = image[:,:,2].astype(float)
    
    # Fórmula de luminância: Y = 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    
    # Máscara para pixels abaixo do threshold
    mask = luminance < threshold
    
    # Magenta é uma cor bem viva e diferente (pode escolher sua cor favorita)
    favorite_color = [255, 0, 255]  # Magenta (ou substitua pela sua cor favorita)
    
    # Aplicar a cor aos pixels abaixo do threshold
    for i in range(3):
        result_channel = result[:,:,i]
        result_channel[mask] = favorite_color[i]
    
    return result


def analyze_histogram(image):
    """Histograma brabo"""

    # Criar figura para o histograma
    plt.figure(figsize=(12, 6))
    
    # Separar os canais de cores
    r_channel = image[:,:,0].flatten()
    g_channel = image[:,:,1].flatten()
    b_channel = image[:,:,2].flatten()
    
    # Definir número de bins e faixa para o histograma
    bins = 256
    pixel_range = (0, 256)
    
    # Plotar histograma para cada canal
    plt.hist(r_channel, bins=bins, range=pixel_range, color='red', alpha=0.6, label='Red')
    plt.hist(g_channel, bins=bins, range=pixel_range, color='green', alpha=0.6, label='Green')
    plt.hist(b_channel, bins=bins, range=pixel_range, color='blue', alpha=0.6, label='Blue')
    
    # Grafico
    plt.xlabel('Intensidade de Pixel (0-255)')
    plt.ylabel('Frequência')
    plt.title('Histograma RGB da Imagem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Meu threshold é a média da intensidade
    threshold = int((np.mean(r_channel) + np.mean(g_channel) + np.mean(b_channel)) / 3)
    
    print(f"Threshold: {threshold}")
    
    return threshold


if __name__ == "__main__":
    images = load_images()
    
    # 1: Juntar as imagens em uma só
    concatenated = concatenate_images(images)
    show_image(concatenated, "1. Imagens coladas")
    
    # 2: Trocar as cores dos pinguins
    colored = change_colors(concatenated)
    show_image(colored, "2. Cores dos pinguins alteradas")
    
    # 3: Espelhar
    flipped = flip_horizontal(colored)
    show_image(flipped, "3. Espelhamento horizontal")
    
    # 4: Recortar um penguin
    cropped = crop_penguin(flipped)
    show_image(cropped, "4. Pinguinzin recortado")
    
    # 5: Histograma e threshold
    threshold = analyze_histogram(cropped)
    #show_image(threshold, "5. Threshold")
    
    # 6: Aplicar cor favorita nos pixeis de baixo do threshold
    final_image = apply_color_threshold(cropped, threshold)
    show_image(final_image, "6. Cor favorita aplicada")