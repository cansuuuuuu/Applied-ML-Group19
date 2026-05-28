import cv2
import tensorflow as tf
import numpy as np

IMAGE_SIZE = (224, 224)
GRAYSCALE_CHANNELS = 1


def resize(data, resize_to=IMAGE_SIZE):
    """
    Resizes the image to a fixed size and converts it to grayscale.
    """
    resized = []
    for img in data:
        img = cv2.resize(img,resize_to)
        resized.append(img)
    return np.array(resized)

def grayify(data):
    """
    Converts the image to grayscale.
    """
    gray = []
    for img in data:
        gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return np.array(gray)



def normalise(data: np.ndarray) -> np.ndarray:
    """
    Normalises pixel values to be between 0 and 1.
    """
    return data.astype("float32") / 255.0   

def reshape_for_cnn(data: np.ndarray) -> np.ndarray:

    """
    Reshapes the data to have a single channel dimension for CNN input.
    """
    return data.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], GRAYSCALE_CHANNELS)   



def load_and_preprocess_image(path: str, label: int) -> tuple:
    """
    Load th images, resizes them and converts it to
    grayscale and normalises pixel values to be between 0 and 1
    """
    raw_image = tf.io.read_file(path)
    image = tf.image.decode_image(raw_image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


augmentation_process = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(15/360), # rotating +- 15 degrees of the image
    tf.keras.layers.RandomBrightness(0.2),  # randomly adding or subtracting brightness to the image (20%)
    tf.keras.layers.RandomContrast(0.2), # randomly increasing or decreasing the contrast of the image (20%)
    tf.keras.layers.RandomZoom(0.1)     # randomly zooming in on the image by up to 10%
])


def augmenting_classes (x, y):

    target_number = 1428 # the number of images all classes should have after augmentation

    classes_to_augment = [0, 1, 6] # the classes that need to be augmented 

    x_balanced = list(x) # the list of images after augmentation, starting with the original images
    y_balanced = list(y) # the list of labels after augmentation, starting with the original labels

    for classes in classes_to_augment:
        class_indices = np.where(y == classes)[0]
        class_images = x[class_indices]
        current_count = len(class_images)
        needed = target_number - current_count

        for i in range(needed):
            source = tf.constant(class_images[i % current_count], dtype=tf.float32)
            augmented = augmentation_process(tf.expand_dims(source, 0))[0]
            augmented = tf.clip_by_value(augmented, 0.0, 1.0)
            x_balanced.append(augmented.numpy())
            y_balanced.append(classes)
    return np.array(x_balanced), np.array(y_balanced)





def apply_clahe(image: tf.Tensor) -> tf.Tensor:

    image_uint8 = tf.cast(image * 255.0, tf.uint8)

    def _equalise(channel):
        hist = tf.histogram_fixed_width(tf.cast(channel, tf.int32), [0, 255], nbins=256)
        cdf = tf.cumsum(hist)
        cdf_min = tf.reduce_min(tf.boolean_mask(cdf, cdf > 0))
        n_pixels = tf.reduce_sum(hist)
        lut = tf.cast(
            tf.round(
                tf.cast(cdf - cdf_min, tf.float32)
                / tf.cast(n_pixels - cdf_min, tf.float32)
                * 255.0
            ),
            tf.uint8,
        )
        return tf.gather(lut, tf.cast(channel, tf.int32))

    flat = tf.reshape(image_uint8, [-1])
    eq = _equalise(flat)
    eq = tf.reshape(tf.cast(eq, tf.float32) / 255.0, tf.shape(image))
    return eq




# def build_train_dataset(
#     image_paths: list,
#     labels: list,
#     batch_size: int = 32,
#     use_clahe: bool = False,
# ) -> tf.data.Dataset:
#     """
#     Creates the training dataset pipeline with preprocessing,
#     augmentation, shuffling and batching.
#     """
#     ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
#     ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
#     if use_clahe:
#         ds = ds.map(
#             lambda img, lbl: (apply_clahe(img), lbl),
#             num_parallel_calls=tf.data.AUTOTUNE,
#         )
#     ds = ds.map(augmentation_process, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds


# def build_val_dataset(image_paths: list,labels: list,batch_size: int = 32,) -> tf.data.Dataset:
#     """
#     Loads and preprocesses validation and test images,
#     then batches the dataset without augmentation.
#     """
#     ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
#     ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds
