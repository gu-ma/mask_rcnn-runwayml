import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import runway
from runway.data_types import image, file

class PastaConfig(Config):
    NAME = "pasta"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + pasta
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7

config = PastaConfig()
config.display()


@runway.setup(options={'weights': file(extension='.h5')})
def setup(opts):
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs')
    model.load_weights(opts['weights'], by_name=True)
    return model


@runway.command('detect', inputs={'image': image}, outputs={'image': image})
def detect(model, inputs):
    img = inputs['image']
    img = np.array(img)
    result = model.detect([img])[0]
    mask = result['masks']
    mask_found = mask.shape[-1] > 0
    if mask_found:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        mask = np.where(mask, [255,255,255], [0,0,0]).astype(np.uint8)
    else:
        mask = img.astype(np.uint8)
    return { 'image': mask }


if __name__ == '__main__':
    runway.run()