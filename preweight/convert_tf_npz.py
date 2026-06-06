import tensorflow as tf
import numpy as np
import os

def convert(model='s'):
    path_to_downloaded_file = tf.keras.utils.get_file(
                origin=f'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-{model}-21k.tgz',
                untar=True)
    print(path_to_downloaded_file)
    ckpt = tf.train.latest_checkpoint(os.path.join(path_to_downloaded_file, f'efficientnetv2-{model}-21k'))

    print(ckpt)
    reader = tf.train.load_checkpoint(ckpt)
    variable_map = reader.get_variable_to_shape_map()

    weights = {}
    for key in variable_map:
        print(key, variable_map[key])
        tensor = reader.get_tensor(key)
        weights[key] = tensor

    np.savez(f'efficientnetv2-{model}-21k', **weights)

if __name__ == '__main__':
    for model in ['s','m','l','xl']:
        print('model:', model)
        convert(model)