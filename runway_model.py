import runway
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils

class PastaConfig(Config):
    NAME = "pasta"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + pasta
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7

config = PastaConfig()
# config.display()

setup_options = {
    'checkpoint': runway.file(extension='.h5'),
    'min_confidence': runway.number(min=0, max=1, step=.1, default=.7),
}

@runway.setup(options=setup_options)
def setup(opts):
    config.DETECTION_MIN_CONFIDENCE = opts['min_confidence']
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs')
    model.load_weights(opts['checkpoint'], by_name=True)
    return model


@runway.command('detect', inputs={'image': runway.image}, outputs={'image': runway.image})
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
    runway.run(model_options={'checkpoint': './checkpoints/mask_rcnn_pasta_0030.h5'})