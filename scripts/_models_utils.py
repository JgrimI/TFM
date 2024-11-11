import numpy as np
import os
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)


def load_datasets(use_augmentation=True):
    """
    Carga los datasets de entrenamiento, validación y prueba desde las rutas especificadas.

    Parámetros:
    -----------
    use_augmentation : bool
        Indica si se deben cargar los datasets con augmentación (True) o sin augmentación (False).

    Retorna:
    --------
    train_dataset, val_dataset, test_dataset : tf.data.Dataset
        Los datasets de entrenamiento, validación y prueba.
    """
    # Determinamos la ruta según si se usa augmentación o no
    if use_augmentation:
        aug_type = "augmented"
    else:
        aug_type = "not_augmented"

    # Definimos las rutas para los datasets
    train_dir = f"../data/tf/{aug_type}/train"
    val_dir = f"../data/tf/{aug_type}/val"
    test_dir = f"../data/tf/{aug_type}/test"

    # Cargamos los datasets usando TensorFlow
    train_data = tf.data.Dataset.load(train_dir)
    val_data = tf.data.Dataset.load(val_dir)
    test_data = tf.data.Dataset.load(test_dir)

    return train_data, val_data, test_data


def load_data_with_prefetch(use_augmentation=True, batch_size=1):
    """
    Carga los datasets de entrenamiento, validación y prueba, aplicando prefetching para optimizar el rendimiento.

    Parámetros:
    -----------
    use_augmentation : bool
        Indica si se deben cargar los datasets con augmentación (True) o sin augmentación (False).
    batch_size : int
        Tamaño del lote para cada dataset.

    Retorna:
    --------
    train_dataset, val_dataset, test_dataset : tf.data.Dataset
        Los datasets de entrenamiento, validación y prueba con prefetching.
    """
    # Determinamos la ruta según si se usa augmentación o no
    aug_type = "augmented" if use_augmentation else "not_augmented"

    # Definimos las rutas para los datasets
    train_dir = f"../data/tf/{aug_type}/train"
    val_dir = f"../data/tf/{aug_type}/val"
    test_dir = f"../data/tf/{aug_type}/test"

    # Cargamos los datasets usando TensorFlow, aplicando batch y prefetch
    train_data = (
        tf.data.experimental.load(train_dir)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_data = (
        tf.data.experimental.load(val_dir).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    test_data = (
        tf.data.experimental.load(test_dir).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )

    return train_data, val_data, test_data


# Función para aplanar las dimensiones del dataset
def reshape_data(image_batch, label_batch):
    """
    Aplana el dataset de imágenes y etiquetas eliminando la dimensión de secuencia.

    Parámetros:
    -----------
    image_batch : tf.Tensor
        Lote de imágenes con dimensiones originales (16, 256, 256, 4).
    label_batch : tf.Tensor
        Lote de etiquetas con dimensiones originales (16, 256, 256, 1).

    Retorna:
    --------
    image_batch : tf.Tensor
        Imágenes reestructuradas a (batch_size*16, 256, 256, 4).
    label_batch : tf.Tensor
        Etiquetas reestructuradas a (batch_size*16, 256, 256, 1).
    """
    # Reestructuramos las imágenes y etiquetas eliminando la dimensión de secuencia
    image_batch = tf.reshape(
        image_batch, (-1, 256, 256, 4)
    )  # De (16, 256, 256, 4) a (batch_size*16, 256, 256, 4)
    label_batch = tf.reshape(
        label_batch, (-1, 256, 256, 1)
    )  # De (16, 256, 256, 1) a (batch_size*16, 256, 256, 1)
    return image_batch, label_batch


def show_and_save_predictions(
    index, test_data, unet_model, output_directory, batch_size=16
):
    """
    Guarda las predicciones de un modelo para un índice específico, incluyendo la imagen de entrada,
    la predicción, la predicción binaria y la verdad de terreno.

    Parámetros:
    -----------
    index : int
        Índice de la imagen que se desea visualizar.
    test_data : tf.data.Dataset
        Dataset de prueba.
    output_directory : str
        Directorio donde se guardarán las predicciones.
    batch_size : int
        Tamaño del lote del dataset (por defecto 16).

    Retorna:
    --------
    None
    """
    # Directorio donde se guardarán las predicciones para el índice dado
    prediction_dir = f"image_{index}"

    # Verificamos si los directorios existen, de lo contrario, los creamos
    if not os.path.exists(
        os.path.join(output_directory, "predictions", prediction_dir)
    ):
        os.makedirs(
            os.path.join(output_directory, "predictions", prediction_dir, "input_image")
        )
        os.makedirs(
            os.path.join(
                output_directory, "predictions", prediction_dir, "ground_truth"
            )
        )
        os.makedirs(
            os.path.join(output_directory, "predictions", prediction_dir, "prediction")
        )
        os.makedirs(
            os.path.join(
                output_directory, "predictions", prediction_dir, "prediction_binary"
            )
        )

    # Creamos un iterador cíclico para recorrer el dataset de prueba
    test_data_iter = iter(itertools.cycle(test_data))

    # Iteramos hasta alcanzar el índice deseado
    for i in range(index + 1):
        image_batch, label_batch = next(test_data_iter)

    # Aplanamos el índice si excede el tamaño del lote
    batch_index = index % batch_size

    # Normalizamos la imagen para visualización
    image = image_batch[batch_index].numpy()
    image_rgb = np.stack(
        (
            (image[:, :, 0] - np.min(image[:, :, 0]))
            * 255.0
            / (np.max(image[:, :, 0]) - np.min(image[:, :, 0])),
            (image[:, :, 1] - np.min(image[:, :, 1]))
            * 255.0
            / (np.max(image[:, :, 1]) - np.min(image[:, :, 1])),
            (image[:, :, 2] - np.min(image[:, :, 2]))
            * 255.0
            / (np.max(image[:, :, 2]) - np.min(image[:, :, 2])),
        ),
        axis=-1,
    ).astype(np.uint8)

    # Hacemos la predicción con el modelo
    prediction = unet_model.predict(np.expand_dims(image, axis=0))[0]

    # Guardamos la imagen de entrada
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "input_image",
            f"{index}.png",
        ),
        image_rgb,
    )

    # Guardamos la verdad de terreno (ground truth)
    ground_truth = label_batch[batch_index].numpy()
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "ground_truth",
            f"{index}.png",
        ),
        np.squeeze(ground_truth),
        cmap="gray",
    )

    # Guardamos la predicción
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "prediction",
            f"{index}.png",
        ),
        np.squeeze(prediction),
        cmap="gray",
    )

    # Convertimos la predicción a binario y la guardamos
    binary_prediction = np.where(prediction > 0.5, 1, 0)
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "prediction_binary",
            f"{index}.png",
        ),
        np.squeeze(binary_prediction),
        cmap="gray",
    )


def save_and_visualize_predictions_fcn(
    index, fcn_model, test_data, output_directory, batch_size=16, show=False
):
    """
    Guarda las predicciones del modelo para un índice dado, incluyendo la imagen de entrada,
    la verdad de terreno, la predicción y la predicción binaria. Opcionalmente, muestra las predicciones
    en una figura con subplots.

    Parámetros:
    -----------
    index : int
        Índice de la imagen en el dataset de prueba.
    fcn_model : tf.keras.Model
        Modelo Fully Convolutional Network (FCN) para hacer las predicciones.
    test_data : tf.data.Dataset
        Dataset de prueba.
    output_directory : str
        Directorio donde se guardarán las predicciones.
    batch_size : int
        Tamaño del lote del dataset (por defecto es 16).
    show : bool
        Si es True, mostrará las predicciones usando matplotlib, además de guardarlas.

    Retorna:
    --------
    None
    """

    # Crear el directorio para guardar las predicciones de una imagen en particular
    prediction_dir = f"image_{index}"
    if not os.path.exists(
        os.path.join(output_directory, "predictions", prediction_dir)
    ):
        os.makedirs(
            os.path.join(output_directory, "predictions", prediction_dir, "input_image")
        )
        os.makedirs(
            os.path.join(
                output_directory, "predictions", prediction_dir, "ground_truth"
            )
        )
        os.makedirs(
            os.path.join(output_directory, "predictions", prediction_dir, "prediction")
        )
        os.makedirs(
            os.path.join(
                output_directory, "predictions", prediction_dir, "prediction_binary"
            )
        )

    # Crear un iterador cíclico para recorrer el dataset de prueba
    test_data_iter = iter(itertools.cycle(test_data))

    # Iteramos hasta alcanzar el índice deseado
    for i in range(index + 1):
        image_batch, label_batch = next(test_data_iter)

    # Aplanamos el índice si excede el tamaño del lote
    batch_index = index % batch_size
    image = image_batch[batch_index].numpy()

    # Normalizar la imagen para visualización
    image_rgb = np.stack(
        (
            (image[:, :, 0] - np.min(image[:, :, 0]))
            * 255.0
            / (np.max(image[:, :, 0]) - np.min(image[:, :, 0])),
            (image[:, :, 1] - np.min(image[:, :, 1]))
            * 255.0
            / (np.max(image[:, :, 1]) - np.min(image[:, :, 1])),
            (image[:, :, 2] - np.min(image[:, :, 2]))
            * 255.0
            / (np.max(image[:, :, 2]) - np.min(image[:, :, 2])),
        ),
        axis=-1,
    ).astype(np.uint8)

    # Hacer predicción con el modelo
    prediction = fcn_model.predict(np.expand_dims(image, axis=0))[0]

    # Guardar la imagen de entrada
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "input_image",
            f"{index}.png",
        ),
        image_rgb,
    )

    # Guardar la verdad de terreno
    ground_truth = label_batch[batch_index].numpy()
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "ground_truth",
            f"{index}.png",
        ),
        np.squeeze(ground_truth),
        cmap="gray",
    )

    # Guardar la predicción
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "prediction",
            f"{index}.png",
        ),
        np.squeeze(prediction),
        cmap="gray",
    )

    # Convertir la predicción a binaria y guardarla
    prediction_binary = np.where(prediction > 0.5, 1, 0)
    plt.imsave(
        os.path.join(
            output_directory,
            "predictions",
            prediction_dir,
            "prediction_binary",
            f"{index}.png",
        ),
        np.squeeze(prediction_binary),
        cmap="gray",
    )

    # Si se desea, mostrar las predicciones en subplots
    if show:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(image_rgb)
        ax[0, 0].set_title("Imagen de entrada")
        ax[0, 1].imshow(np.squeeze(ground_truth), cmap="gray")
        ax[0, 1].set_title("Verdad de terreno")
        ax[1, 0].imshow(np.squeeze(prediction), cmap="gray")
        ax[1, 0].set_title("Predicción")
        ax[1, 1].imshow(np.squeeze(prediction) > 0.5, cmap="gray")
        ax[1, 1].set_title("Predicción (binaria)")
        # Ocultar los ejes en todas las subplots
        for i in range(2):
            for j in range(2):
                ax[i, j].axis("off")

        # Mostrar la figura con las predicciones
        plt.show()


def evaluate_model(
    description,
    test_data,
    model,
    input_shape,
    is_shuffled,
    batch_size,
    epochs_num,
    augment_settings,
    threshold=0.5,
):
    """
    Evalúa el modelo con el dataset de prueba, calculando métricas de rendimiento.

    Parámetros:
    -----------
    description : str
        Descripción del modelo o experimento.
    test_data : tf.data.Dataset
        Dataset de prueba.
    model : tf.keras.Model
        Modelo a evaluar.
    input_shape : tuple
        Dimensiones de entrada del modelo.
    is_shuffled : bool
        Si los datos fueron mezclados o no.
    batch_size : int
        Tamaño del lote para la evaluación.
    epochs_num : int
        Número de épocas usadas en el entrenamiento.
    augment_settings : dict
        Configuración de augmentación aplicada.
    threshold : float
        Umbral de decisión para las predicciones del modelo.

    Retorna:
    --------
    model_info : dict
        Diccionario con la información del modelo y las métricas calculadas.
    """
    # Inicializamos listas para almacenar las etiquetas verdaderas y las predicciones
    true_labels = []
    predicted_labels = []

    # Iteramos sobre el dataset de prueba y hacemos predicciones
    for images, labels in test_data:
        preds = model.predict(images)

        for i in range(preds.shape[0]):
            # Aplanamos las etiquetas verdaderas y las predicciones
            true_flat = np.squeeze(labels[i].numpy()).flatten()
            pred_flat = np.squeeze(preds[i]).flatten()

            true_labels.append(true_flat)
            predicted_labels.append(pred_flat > threshold)

    # Concatenamos las etiquetas y predicciones para calcular las métricas
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)

    # Calculamos las métricas
    accuracy = accuracy_score(true_labels, predicted_labels)
    iou_score = jaccard_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    # Guardamos la información del modelo y las métricas en un diccionario
    model_metrics = {
        "description": [description],
        "date_saved": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "input_shape": [input_shape],
        "batch_size": [batch_size],
        "epochs": [epochs_num],
        "shuffled": [is_shuffled],
        "augment_settings": [augment_settings],
        "accuracy": [accuracy],
        "iou": [iou_score],
        "f1_score": [f1],
        "precision": [precision],
        "recall": [recall],
    }

    return model_metrics
