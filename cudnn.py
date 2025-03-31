import tensorflow as tf
import numpy as np

# test tensorflow

print(tf.__version__)
print(tf.test.is_gpu_available())
# cudnn
print(tf.test.is_built_with_cuda())
