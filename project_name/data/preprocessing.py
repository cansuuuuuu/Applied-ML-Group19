import tensorflow as tf

IMAGE_SIZE = (512, 512)
GRAYSCALE_CHANNELS = 1

_rotator: tf.keras.layers.Layer = tf.keras.layers.RandomRotation(
    factor=15 / 360,
    fill_mode="reflect",
    interpolation="bilinear",
)


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


def augment(image: tf.Tensor, label: int) -> tuple:
    """
    These are the augmentations we will apply to the training images:
      - Horizontal flip 
      - Brightness, contrast and saturation adjustments
      - Small random rotation (±15°) 
    """
    image = tf.image.random_flip_left_right(image)

    image_rgb = tf.image.grayscale_to_rgb(image)
    image_rgb = tf.image.random_brightness(image_rgb, max_delta=0.2)
    image_rgb = tf.image.random_contrast(image_rgb, lower=0.8, upper=1.2)
    image_rgb = tf.image.random_saturation(image_rgb, lower=0.8, upper=1.2)
    image = tf.image.rgb_to_grayscale(image_rgb)

    image = _random_rotate(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


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


def _random_rotate(image: tf.Tensor) -> tf.Tensor:
    """
    Rotates the image by a random angle of +/- 15 degrees.
    Since it expects a batch of images, we add a batch dimension, 
    apply the rotation and then remove it again. 
    """
    return _rotator(tf.expand_dims(image, 0))[0]


def build_train_dataset(
    image_paths: list,
    labels: list,
    batch_size: int = 32,
    use_clahe: bool = False,
) -> tf.data.Dataset:
    """
    Creates the training dataset pipeline with preprocessing,
    augmentation, shuffling and batching.
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if use_clahe:
        ds = ds.map(
            lambda img, lbl: (apply_clahe(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_val_dataset(image_paths: list,labels: list,batch_size: int = 32,) -> tf.data.Dataset:
    """
    Loads and preprocesses validation and test images,
    then batches the dataset without augmentation.
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
