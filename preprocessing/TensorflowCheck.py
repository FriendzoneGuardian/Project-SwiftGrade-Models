import tensorflow as tf
from tensorflow.python.client import timeline
import GPUtil
        
print("TensorFlow Version:", tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

gpus = GPUtil.getGPUs()
if gpus:
    print("Detected GPU(s):")
    for gpu in gpus:
        print(f"  {gpu.name} ({gpu.memoryTotal}MB, Load: {gpu.load*100:.1f}%)")
else:
    print("No GPUs detected")

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)