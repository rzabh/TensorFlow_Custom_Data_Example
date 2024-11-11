import os
import tensorflow as tf

class ObjectDetectionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the object detection model class.

        Args:
            input_shape (tuple): Input shape for the model.
            num_classes (int): Number of classes for classification.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    @staticmethod
    def parse_label_file(label_path):
        """
        Parse a label file into class IDs and bounding boxes.

        Args:
            label_path (str): Path to the label file.

        Returns:
            list: List of tuples containing class ID and bounding box.
        """
        with open(label_path, 'r') as f:
            labels = []
            for line in f:
                values = line.strip().split()
                class_id = float(values[0])
                bbox = list(map(float, values[1:]))  # [x, y, width, height]
                labels.append((class_id, bbox))
        return labels

    def load_image_and_labels(self, image_path, label_path):
        """
        Load and preprocess an image and its corresponding labels.

        Args:
            image_path (str): Path to the image file.
            label_path (str): Path to the label file.

        Returns:
            tuple: Preprocessed image and labels (class IDs and bounding boxes).
        """
        # Load and preprocess the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.input_shape[:2])

        # Wrap parse_label_file to work with TensorFlow
        labels = tf.numpy_function(self.parse_label_file, [label_path], [tf.float32, tf.float32])

        # Reshape labels for consistency
        labels[0].set_shape([None])  # class_ids
        labels[1].set_shape([None, 4])  # bboxes

        return image, (labels[0], labels[1])

    def prepare_dataset(self, image_dir, label_dir, batch_size=32):
        """
        Prepare a TensorFlow dataset for the specified image and label directories.

        Args:
            image_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.
            batch_size (int): Batch size for the dataset.

        Returns:
            tf.data.Dataset: Prepared dataset.
        """
        # List image and label files
        image_files = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ])
        label_files = sorted([
            os.path.join(label_dir, fname.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            for fname in os.listdir(image_dir)
        ])

        # Load image and label pairs
        dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
        dataset = dataset.map(lambda x, y: self.load_image_and_labels(x, y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def create_model(self):
        """
        Create a TensorFlow object detection model.

        Returns:
            tf.keras.Model: The compiled object detection model.
        """
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False)
        base_model.trainable = False  # Freeze base model weights

        # Add custom layers for object detection
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Output layers
        bbox_output = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(x)
        class_output = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='class')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=[class_output, bbox_output])
        return model
