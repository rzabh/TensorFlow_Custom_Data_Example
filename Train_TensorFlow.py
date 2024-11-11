import os
import tensorflow as tf
import tf2onnx
from Utils_Tensorflow.utils_tensorflow import ObjectDetectionModel 

# Directories
# Define directories
file_path = os.path.dirname(__file__)


image_dirs = {
    "train": os.path.join(file_path, 'datasets', 'coco8', 'images', 'train'),
    "test": os.path.join(file_path, 'datasets', 'coco8', 'images', 'test'),
    "val": os.path.join(file_path, 'datasets', 'coco8', 'images', 'val')
}

label_dirs = {
    "train": os.path.join(file_path, 'datasets', 'coco8', 'labels', 'train'),
    "test": os.path.join(file_path, 'datasets', 'coco8', 'labels', 'test'),
    "val": os.path.join(file_path, 'datasets', 'coco8', 'labels', 'val')
}

# Instantiate the ObjectDetectionModel class
od_model = ObjectDetectionModel(input_shape=(224, 224, 3), num_classes=14)

# Prepare datasets
train_dataset = od_model.prepare_dataset(image_dirs['train'], label_dirs['train'])
test_dataset = od_model.prepare_dataset(image_dirs['test'], label_dirs['test'])
val_dataset = od_model.prepare_dataset(image_dirs['val'], label_dirs['val']) if os.path.exists(image_dirs['val']) else None

# Create and summarize the model
model = od_model.create_model()
model.summary()

onnx_model_path = os.path.join(file_path,'Runs_TensorFlow','exported_model.onnx')
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)  # Adjust the input shape and dtype if needed
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_model_path)

print(f"Model exported to ONNX format at: {onnx_model_path}")
