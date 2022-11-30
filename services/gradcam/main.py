import logging
import time

import numpy as np
import tensorflow as tf
import uvicorn
import math
from starlette.applications import Starlette
from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from tensorflow import keras

from constants import IMAGENET_CLASSES

logging.basicConfig(level=logging.INFO)

model_builder = keras.applications.xception.Xception
model = model_builder(weights="imagenet")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


class Healthcheck(HTTPEndpoint):
    def get(self, request):
        return JSONResponse({"ok": True})


class Heatmap(HTTPEndpoint):
    async def post(self, request):
        tic = time.perf_counter()
        request_content = Request(self.scope, self.receive)
        try:
            request_content = await request_content.json()
        except RuntimeError:
            request_content = "Receive channel not available"

        # Size: (299, 299, 3) | Pixels: [0 ... 255] | Type: python list
        img_array = request_content["img_array"]

        # Size: (299, 299, 3) | Pixels: [0 ... 255] | Type: numpy array
        img_array = np.array(img_array)

        # Size: (1, 299, 299, 3) | Pixels: [0 ... 255] | Type: numpy array
        img_array = np.expand_dims(img_array, axis=0)

        # Size: (1, 299, 299, 3) | Pixels: [0 ... 1] | Type: numpy array
        img_array_normalized = keras.applications.xception.preprocess_input(img_array)

        pred_class = request_content["img_class"]
        pred_index = IMAGENET_CLASSES[pred_class]
        last_conv_layer_name = "block14_sepconv2_act"
        heatmap = make_gradcam_heatmap(
            img_array_normalized, model, last_conv_layer_name, pred_index
        )
        heatmap[np.isnan(heatmap)] = 0

        template_response = {"data": heatmap.tolist(), "meta": {}}

        response = JSONResponse(template_response)

        toc = time.perf_counter()
        logging.info(f" Heatmap data in {toc - tic:0.4f} seconds")

        return response


routes = [
    Route("/heatmap", Heatmap),
    Route("/healthcheck", Healthcheck),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2)
