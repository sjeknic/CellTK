from typing import Tuple

import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, UpSampling1D,
                                     Conv2D, MaxPooling2D, UpSampling2D,
                                     BatchNormalization, Input, Activation,
                                     Concatenate)
from tensorflow.keras import backend as K
from keras.layers.advanced_activations import LeakyReLU

from base.utils import Image


ALPHA = 0.3
UNET_KWS = dict(classes=3, kernel_size=(3, 3), steps=2, layers=2,
                init_filters=64, transfer=True, activation='relu',
                padding='same', mode=0, momentum=0.9)


class UNetModel():
    """
    TODO:
        - Add option to save and load models from JSON
        - Add option to pass custom models
        - Add custom load weights function
        - Should there be a base model class and then subclasses for UNet v UPeak?
    """
    def __init__(self,
                 dimensions: Tuple[int],
                 weight_path: str,
                 model: str = 'unet'
                 ) -> None:
        # Get model options and build model
        if model == 'unet':
            kws = UNET_KWS
        elif model == 'upeak':
            kws = UPEAK_KWS
        else:
            raise ValueError(f'Did not find model {model}.')

        self.model = self._build_model(dimensions, **kws)
        self.model.load_weights(weight_path)

    def predict(self,
                image: Image = None
                ):
        # First, image has to be normalized
        return self.model.predict(image)

    def _build_model(self,
                     dimensions: Tuple[int],
                     classes: int = 3,
                     steps: int = 2,
                     layers: int = 2,
                     kernel: (int, tuple) = (3, 3),
                     strides: int = 1,
                     init_filters: int = 64,
                     activation: str = 'relu',
                     transfer: bool = True,
                     padding: str = 'valid',
                     mode: int = 0,
                     momentum: float = 0.9,
                     **kwargs
                     ) -> tensorflow.keras.models.Model:

        # Select the appropriate layers
        if len(dimensions) == 2:
            # 1D inputs
            conv_layer = Conv1D
            pool_layer = MaxPooling1D
            up_layer = UpSampling1D
        elif len(dimensions) == 3:
            # 2D inputs
            conv_layer = Conv2D
            pool_layer = MaxPooling2D
            up_layer = UpSampling2D
        else:
            raise ValueError(f'Model can not be built for dimensions {dimensions}.')

        # Add LeakyRelU as an option
        if activation == 'leaky':
            activation = lambda x: LeakyReLU(alpha=ALPHA)(x)

        # TODO: Any args that change need to be passed around
        def _add_pool_module(input_layer,
                             init_filter: int = 64):
            trans_layers = []
            filt = init_filter
            for _ in range(steps):
                # Build a convolutional model and pool it
                input_layer = _add_conv_module(input_layer, filt)

                # Handle any layers that are transfered
                if transfer:
                    trans_layers.append(input_layer)

                input_layer = pool_layer()(input_layer)

                # More output features as pooling progresses
                filt *= 2

            return input_layer, trans_layers

        def _add_up_module(input_layer,
                           init_filter: int = 256,
                           trans_layers: list = []):
            # trans_layers starts at the top of the model,
            # we are the bottom when this module starts
            trans_layers = trans_layers[::-1]

            filt = init_filter
            for s in range(steps):
                input_layer = up_layer()(input_layer)

                if transfer:
                    input_layer = Concatenate()([input_layer, trans_layers[s]])

                input_layer = _add_conv_module(input_layer, filt)

                # Fewer output features as upsampling progresses
                filt = int(filt) * 0.5

            return input_layer

        def _add_conv_module(input_layer, filt: int = 64):
            '''
            Build successive Conv-BatchNorm layers
            '''
            for _ in range(layers):
                input_layer = conv_layer(filters=filt, kernel_size=kernel,
                                         strides=strides, activation=activation,
                                         padding=padding)(input_layer)
                input_layer = BatchNormalization()(input_layer)

            return input_layer

        # TODO: Will this break other models?
        K.clear_session()

        x = Input(dimensions)

        # Pooling
        y, trans_layers = _add_pool_module(x)

        # Base layer
        y = _add_conv_module(y)

        # Upsampling
        filt = y.shape[-1]
        y = _add_up_module(y, filt, trans_layers)

        # Output
        y = conv_layer(classes, 1)(y)
        y = Activation('softmax')(y)

        return tensorflow.keras.models.Model(x, y)





