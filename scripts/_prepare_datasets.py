import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import rasterio


def prepare_datasets(
    batch_size=16,
    shuffle_data=True,
    augment_options=None,
    img_size=256,
    data_path="../data/edit/",
):
    """
    Función para procesar datos y crear datasets de entrenamiento, validación y prueba.

    Esta función lee los archivos de imágenes y etiquetas, aplica posibles
    aumentaciones, y crea datasets de TensorFlow listos para el entrenamiento.

    Parámetros:
    -----------
    batch_size : int
        Tamaño del lote para el dataset.
    shuffle_data : bool
        Si se deben mezclar los datos en cada época.
    augment_options : dict, opcional
        Diccionario con las configuraciones para la augmentación.
    img_size : int
        Tamaño (ancho y alto) de las imágenes.
    data_path : str
        Ruta relativa al directorio de datos.

    Retorna:
    --------
    train_dataset, val_dataset, test_dataset : tf.data.Dataset
        Datasets de entrenamiento, validación y prueba.
    image_paths, label_paths : list
        Listas de las rutas de imágenes y etiquetas para cada dataset.
    """

    def load_image(filepath):
        """
        Lee una imagen GeoTIFF desde una ruta y la transforma en un array numpy.

        Parámetros:
        -----------
        filepath : str
            Ruta del archivo de imagen GeoTIFF.

        Retorna:
        --------
        np.ndarray
            Imagen leída y transpuesta, con dimensiones (altura, ancho, canales).
        """
        filepath = filepath.decode("utf-8")
        with rasterio.open(filepath) as src:
            img_data = src.read()
        img_data = np.transpose(img_data, (1, 2, 0))  # Transponemos para (altura, ancho, canales)
        return img_data.astype(np.float32)

    def _parse_data(image_path, label_path):
        """
        Parsea una imagen y una etiqueta usando tf.py_function para leerlas como arrays de TensorFlow.

        Parámetros:
        -----------
        image_path : str
            Ruta de la imagen.
        label_path : str
            Ruta de la etiqueta.

        Retorna:
        --------
        tf.Tensor
            Tensores de imagen y etiqueta.
        """
        image = tf.py_function(
            lambda img_path: load_image(img_path.numpy()), [image_path], tf.float32
        )
        label = tf.py_function(
            lambda lbl_path: load_image(lbl_path.numpy()), [label_path], tf.float32
        )
        return image, label

    def apply_augmentations(image, label, augment_tensor):
        """
        Aplica augmentaciones a una imagen y su etiqueta según las configuraciones definidas en un tensor.

        Parámetros:
        -----------
        image : tf.Tensor
            Imagen a la que se le aplicarán augmentaciones.
        label : tf.Tensor
            Etiqueta correspondiente a la imagen.
        augment_tensor : tf.Tensor
            Tensor que contiene las probabilidades de augmentación.

        Retorna:
        --------
        image, label : tf.Tensor
            Imagen y etiqueta con las augmentaciones aplicadas.
        """
        if augment_options is None:
            return image, label

        # Extraemos probabilidades de augmentación del tensor
        (
            flip_horizontal_prob,
            flip_vertical_prob,
            blur_prob,
            noise_prob,
            brightness_prob,
            contrast_prob,
        ) = tf.unstack(augment_tensor)

        if blur_prob > tf.random.uniform((), maxval=1):
            image = tfa.image.gaussian_filter2d(image, sigma=0.8)

        if brightness_prob > tf.random.uniform((), maxval=1):
            image = tf.image.adjust_brightness(image, delta=0.1)

        if contrast_prob > tf.random.uniform((), maxval=1):
            image = tf.image.adjust_contrast(image, contrast_factor=0.1)

        image = tf.cast(image, dtype=tf.float32)
        label = tf.cast(label, dtype=tf.float32)

        return image, label

    def process_image_and_label(image, label, augment_options):
        """
        Preprocesa una imagen y su etiqueta aplicando augmentaciones (si están definidas).

        Parámetros:
        -----------
        image : tf.Tensor
            Imagen a preprocesar.
        label : tf.Tensor
            Etiqueta correspondiente.
        augment_options : dict, opcional
            Diccionario con las opciones de augmentación a aplicar.

        Retorna:
        --------
        image, label : tf.Tensor
            Imagen y etiqueta preprocesadas y ajustadas a la forma correcta.
        """
        if augment_options:
            augment_tensor = tf.convert_to_tensor(
                list(augment_options.values()), dtype=tf.float32
            )
            image, label = apply_augmentations(image, label, augment_tensor)

        image.set_shape((img_size, img_size, 4))
        label.set_shape((img_size, img_size, 1))

        return image, label

    def build_dataset(image_files, label_files, batch_size, shuffle_data, augment_options):
        """
        Construye un dataset de TensorFlow a partir de archivos de imágenes y etiquetas.

        Parámetros:
        -----------
        image_files : list
            Lista de rutas de los archivos de imagen.
        label_files : list
            Lista de rutas de los archivos de etiqueta.
        batch_size : int
            Tamaño de lote para el dataset.
        shuffle_data : bool
            Si se deben mezclar los datos en el dataset.
        augment_options : dict, opcional
            Diccionario con las configuraciones para las augmentaciones.

        Retorna:
        --------
        tf.data.Dataset
            Dataset listo para el entrenamiento o evaluación.
        """
        image_paths = tf.convert_to_tensor(image_files, dtype=tf.string)
        label_paths = tf.convert_to_tensor(label_files, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.map(_parse_data, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda img, lbl: process_image_and_label(img, lbl, augment_options),
            tf.data.experimental.AUTOTUNE,
        )

        if shuffle_data:
            dataset = dataset.shuffle(buffer_size=len(image_files))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    # Definir los directorios de las imágenes y etiquetas
    project_directory = os.path.dirname(__file__) + "/"

    train_img_dir_amazon = project_directory + data_path + "AMAZON/Training/image/"
    train_lbl_dir_amazon = project_directory + data_path + "AMAZON/Training/label/"

    val_img_dir_amazon = project_directory + data_path + "AMAZON/Validation/images/"
    val_lbl_dir_amazon = project_directory + data_path + "AMAZON/Validation/masks/"

    test_img_dir_amazon = project_directory + data_path + "AMAZON/Test/image/"
    test_lbl_dir_amazon = project_directory + data_path + "AMAZON/Test/mask/"

    # Obtener las rutas de los archivos .tif
    train_img_paths_amazon = [
        os.path.join(train_img_dir_amazon, file)
        for file in os.listdir(train_img_dir_amazon)
        if file.endswith(".tif")
    ]
    train_lbl_paths_amazon = [
        os.path.join(train_lbl_dir_amazon, file)
        for file in os.listdir(train_lbl_dir_amazon)
        if file.endswith(".tif")
    ]

    val_img_paths_amazon = [
        os.path.join(val_img_dir_amazon, file)
        for file in os.listdir(val_img_dir_amazon)
        if file.endswith(".tif")
    ]
    val_lbl_paths_amazon = [
        os.path.join(val_lbl_dir_amazon, file)
        for file in os.listdir(val_lbl_dir_amazon)
        if file.endswith(".tif")
    ]

    test_img_paths_amazon = [
        os.path.join(test_img_dir_amazon, file)
        for file in os.listdir(test_img_dir_amazon)
        if file.endswith(".tif")
    ]
    test_lbl_paths_amazon = [
        os.path.join(test_lbl_dir_amazon, file)
        for file in os.listdir(test_lbl_dir_amazon)
        if file.endswith(".tif")
    ]

    # Crear los datasets de entrenamiento, validación y prueba
    train_dataset = build_dataset(
        train_img_paths_amazon,
        train_lbl_paths_amazon,
        batch_size=batch_size,
        shuffle_data=shuffle_data,
        augment_options=augment_options,
    )
    val_dataset = build_dataset(
        val_img_paths_amazon,
        val_lbl_paths_amazon,
        batch_size=batch_size,
        shuffle_data=False,  # No es necesario mezclar el dataset de validación
        augment_options=None,
    )
    test_dataset = build_dataset(
        test_img_paths_amazon,
        test_lbl_paths_amazon,
        batch_size=batch_size,
        shuffle_data=False,  # No es necesario mezclar el dataset de prueba
        augment_options=None,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_img_paths_amazon,
        train_lbl_paths_amazon,
        val_img_paths_amazon,
        val_lbl_paths_amazon,
        test_img_paths_amazon,
        test_lbl_paths_amazon,
    )
