import tensorflow as tf
import numpy as np

class GradCAM:

    def __init__(self, model, class_idx):

        self.model = model
        self.class_idx = class_idx

        # base MobileNetV2
        self.base_model = model.get_layer(
            'mobilenetv2_1.00_224'
        )

        # last conv
        self.last_conv_layer = self.base_model.get_layer(
            'Conv_1'
        )

        # classifier head
        self.classifier = tf.keras.Sequential([
            layer for layer in model.layers[2:]
        ])

    def compute_heatmap(
        self,
        img_array
    ):

        img_tensor=tf.convert_to_tensor(
            img_array,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            # forward MobileNetV2
            conv_outputs=self.base_model(
                img_tensor,
                training=False
            )

            tape.watch(
                conv_outputs
            )

            # classifier
            predictions=self.classifier(
                conv_outputs
            )

            loss=predictions[
                :,
                self.class_idx
            ]

        grads=tape.gradient(
            loss,
            conv_outputs
        )

        pooled_grads=tf.reduce_mean(
            grads,
            axis=(0,1,2)
        )

        conv_outputs=conv_outputs[0]

        heatmap=tf.reduce_sum(
            pooled_grads*conv_outputs,
            axis=-1
        )

        heatmap=tf.maximum(
            heatmap,
            0
        )

        heatmap/=(
            tf.reduce_max(
                heatmap
            )+1e-8
        )

        return heatmap.numpy()