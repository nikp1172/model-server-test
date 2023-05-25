import os
import io
import base64
import json
import logging
from abc import ABC

import torch
import torch.nn.functional as F
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from transforms import transform

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] -- %(message)s')
logger = logging.getLogger(__name__)
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)

class CNNCIFAR10Handler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        has_gpu = torch.cuda.is_available() and properties.get("gpu_id") is not None
        self.map_location = "cuda" if has_gpu else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if has_gpu else self.map_location
        )
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        model_file = self.manifest["model"].get("modelFile", "")
        logger.debug("Loading eager model without state dict")
        self.model = torch.load(model_pt_path, map_location=self.map_location)
        self.model.to(self.device)
        self.model.eval()
        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        images = []
        logger.debug(data)
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)
            image = transform(image)
            images.append(image)

        images = torch.stack(images).to(self.device)
        return images

    def postprocess(self, data):
        data = F.softmax(data, dim=1).cpu().numpy().tolist()  # B x C
        return [dict(zip(CLASSES, probs)) for probs in data]
