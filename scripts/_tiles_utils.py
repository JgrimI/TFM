import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Función que divide y rellena una imagen con ceros para ajustarse a un tamaño de bloque específico
def segment_and_pad(img, tgt_height, tgt_width):
    """
    Divide y rellena una imagen para ajustarla a un tamaño específico de bloques.

    Esta función toma una imagen en forma de matriz de 3D (bandas, altura, anchura),
    la rellena con ceros si es necesario para ajustarse a las dimensiones objetivo,
    y luego la divide en bloques (tiles) del tamaño deseado.

    Parámetros:
    -----------
    img : numpy.ndarray
        Imagen original con las dimensiones (bandas, altura, anchura).
    tgt_height : int
        Altura objetivo de los bloques en los que se dividirá la imagen.
    tgt_width : int
        Ancho objetivo de los bloques en los que se dividirá la imagen.

    Retorna:
    --------
    list
        Lista de bloques (tiles) de la imagen, cada uno con el tamaño especificado.
    """
    # Extraemos las dimensiones de la imagen (bandas, alto, ancho)
    layers, img_height, img_width = img.shape

    # Calculamos el nuevo alto y ancho, ajustados al tamaño objetivo, agregando relleno si es necesario
    adj_height = int(np.ceil(img_height / tgt_height) * tgt_height)
    adj_width = int(np.ceil(img_width / tgt_width) * tgt_width)

    # Rellenamos la imagen con ceros (modo 'constant') para que se ajuste exactamente a los tamaños objetivo
    adjusted_img = np.pad(
        img,
        ((0, 0), (0, adj_height - img_height), (0, adj_width - img_width)),
        mode="constant",
    )

    # Calculamos cuántos bloques (tiles) se necesitarán tanto en altura como en anchura
    vertical_tiles = adj_height // tgt_height
    horizontal_tiles = adj_width // tgt_width

    # Inicializamos una lista para almacenar los bloques generados
    segmented_pieces = []

    # Dividimos la imagen en bloques
    for v in range(vertical_tiles):
        for h in range(horizontal_tiles):
            # Extraemos el bloque correspondiente de la imagen ajustada
            sub_img = adjusted_img[
                :,
                v * tgt_height : (v + 1) * tgt_height,
                h * tgt_width : (h + 1) * tgt_width,
            ]
            segmented_pieces.append(sub_img)  # Añadimos el bloque a la lista
    return segmented_pieces  # Devolvemos la lista de bloques generados


# Función para guardar los bloques de imagen en archivos separados
def store_segments(segments, output_path, base_label):
    """
    Guarda los bloques generados de una imagen en archivos GeoTIFF.

    Esta función toma una lista de bloques (tiles) de una imagen, y los guarda
    como archivos individuales en formato TIFF en el directorio especificado.

    Parámetros:
    -----------
    segments : list
        Lista de bloques (tiles) de la imagen generados por la función segment_and_pad.
    output_path : str
        Directorio donde se guardarán los archivos.
    base_label : str
        Nombre base para los archivos. Los archivos se numerarán automáticamente.

    Retorna:
    --------
    None
        No retorna nada, guarda los archivos en el disco.
    """
    # Verificamos si la carpeta de salida existe, y si no, la creamos
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iteramos sobre los bloques generados para guardarlos en archivos
    for idx, seg in enumerate(segments):
        # Definimos el nombre del archivo para cada bloque, incluyendo el índice
        seg_file = os.path.join(output_path, f"{base_label}_part_{idx}.tif")

        # Extraemos las dimensiones del bloque (bandas, alto, ancho)
        channels, seg_height, seg_width = seg.shape

        # Abrimos un archivo GeoTIFF para escribir los datos del bloque
        with rasterio.open(
            seg_file,
            "w",
            driver="GTiff",
            height=seg_height,
            width=seg_width,
            count=channels,
            dtype=seg.dtype,
        ) as file:
            # Escribimos cada banda del bloque en el archivo correspondiente
            for layer in range(channels):
                file.write(seg[layer], layer + 1)


def show_geotiff(image_path, mask_path=None, save_path=None):
    """
    Visualiza una imagen GeoTIFF y su máscara opcional.

    Esta función lee una imagen GeoTIFF, la normaliza y la visualiza.
    Si se proporciona una máscara, también la muestra junto a la imagen.
    Además, permite guardar la visualización como un archivo si se especifica una ruta de guardado.

    Parámetros:
    -----------
    image_path : str
        Ruta al archivo de imagen GeoTIFF que se desea visualizar.
    mask_path : str, opcional
        Ruta al archivo de máscara GeoTIFF que se desea visualizar junto a la imagen (por defecto es None).
    save_path : str, opcional
        Ruta donde se guardará la visualización como archivo de imagen (por defecto es None).

    Retorna:
    --------
    None
        Muestra o guarda la visualización de la imagen y, opcionalmente, la máscara.
    """
    try:
        # Leer los datos de la imagen
        with rasterio.open(image_path) as src:
            image_data = src.read().astype(np.float32)
    except Exception as e:
        print(f"Error leyendo la imagen: {e}")
        return

    # Verificamos si tiene suficientes bandas (RGB)
    if image_data.shape[0] < 3:
        print("La imagen no tiene suficientes bandas para visualizar en RGB.")
        return

    # Normalización de las bandas RGB para visualización (cada banda se escala
    # a un rango de 0-255)
    def normalize_band(band):
        """Normaliza una banda de imagen entre 0 y 255."""
        min_val = np.min(band)
        max_val = np.max(band)
        return np.clip((band - min_val) * 255.0 / (max_val - min_val), 0, 255)

    image_rgb = np.stack(
        (
            normalize_band(image_data[0]),
            normalize_band(image_data[1]),
            normalize_band(image_data[2]),
        ),
        axis=-1,
    ).astype(np.uint8)

    # Leer la máscara si se proporciona
    if mask_path:
        try:
            with rasterio.open(mask_path) as src:
                mask_data = src.read(1)
        except Exception as e:
            print(f"Error leyendo la máscara: {e}")
            return

        # Mostrar imagen y máscara en subplots
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(image_rgb)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(mask_data, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

    # Mostrar solo la imagen si no se proporciona la máscara
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.title("Image")
        plt.axis("off")

    # Guardar en disco si se proporciona una ruta de guardado
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def check_dimensions(folder_path):
    """
    Revisa las dimensiones de los archivos GeoTIFF en una carpeta y guarda los resultados en un archivo CSV.

    Parámetros:
    -----------
    folder_path : str
        Ruta a la carpeta que contiene los archivos .tif para revisar.

    Retorna:
    --------
    None
        Los resultados se guardan en un archivo CSV, sin retorno.
    """
    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".tif")
    ]

    # Para cada archivo, obtenemos las dimensiones y el número de canales
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            width, height = src.width, src.height
            channels = src.count

        # Creamos un DataFrame temporal y lo guardamos en el archivo CSV
        pd.DataFrame(
            [[file_path, f"({width}, {height})", channels]],
            columns=["File", "Dimensions", "Channels"],
        ).to_csv(
            "../data/edit/generated_tile_shape_diagnostics.csv",
            mode="a",
            header=False,
            index=False,
        )
