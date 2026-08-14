import tensorflow as tf
print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("CUDA available:", len(tf.config.list_physical_devices("GPU")) > 0)
