from typing import Tuple, List

import numpy as np
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, UpSampling1D,
                                     Conv2D, MaxPooling2D, UpSampling2D,
                                     BatchNormalization, Input, Activation,
                                     Concatenate)
from tensorflow.keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import tensorflow.keras.models

from base.utils import Image


ALPHA = 0.3
# TODO: Write docstring explaining kwargs
UNET_KWS = dict(classes=3, kernel_size=(3, 3), steps=3, layers=2,
                init_filters=64, transfer=True, activation='relu',
                padding='same', mode=0, momentum=0.9)


class UNetModel():
    """
    TODO:
        - Add option to save and load models from JSON
        - Add option to pass custom models
        - Add custom load weights function
        - Expand to work with multiple types of models (UNet, UPeak, etc.)
        - Should there be a base model class and then subclasses for UNet v UPeak?
    """
    def __init__(self,
                 dimensions: Tuple[int],
                 weight_path: str,
                 model: str = 'unet'
                 ) -> None:
        # Build model
        self.model_kws = UNET_KWS
        adj_dims = self.pad_model_dimension(dimensions, self.model_kws['steps'])
        self.model = self._build_model(adj_dims, **self.model_kws)
        self.model.load_weights(weight_path)

    def predict(self,
                image: Image = None
                ) -> Image:
        """
        Assume that image will always be 3D, even if only one frame is received
        """
        # First, image has to be normalized
        orig_shape = image.shape

        # This is the expected dimension, add axis at -1 for channels
        if image.ndim == 3:
            image = np.expand_dims(image, -1)
        else:
            raise ValueError(f'Expected 3D stack, got {image.ndim}')
        image, pads = self.pad_2d(image, self.model_kws['steps'])

        return self.model.predict(image)

    def pad_model_dimension(self, shape: Tuple[int], model_steps: int) -> Tuple[int]:
        """
        """
        # Dimensions are independent of frames. height x width x channels(1)
        pads = self._calculate_pads(shape[:-1], 2 ** model_steps)
        # Remember to account for the channels dimension
        pads.append((0, 0))
        # Pads is a list of tuples that sums to the delta on each axis
        return [int(s + sum(p)) for s, p in zip(shape, pads)]

    def pad_2d(self,
               image: Image,
               model_steps: int,
               mode: str = 'constant',
               constant_values: float = 0.
               ) -> Image:
        """
        This assumes that image is supposed to be 4 dimensional.
        Batch x Height x Width x Channels
        """
        pads = self._calculate_pads(image.shape, 2 ** model_steps)
        # The channel and batch axes will not be padded
        pads[0] = (0, 0)
        pads[-1] = (0, 0)

        return np.pad(image, pads, mode=mode, constant_values=constant_values), pads

    def _calculate_pads(self, shape: Tuple[int], target_mod: int) -> List[Tuple]:
        """
        """
        axes_deltas = [target_mod - (s % target_mod) for s in shape]

        if not all(a == 0 for a in axes_deltas):
            # Divide by two because padding both sides
            pads = [int(a / 2.) for a in axes_deltas]
            # Adjust for odds
            pads = [(p, p + 1) if p % 2 else (p, p) for p in pads]
        else:
            pads = [(0, 0) * len(axes_deltas)]

        return pads

    '''TODO: I think this is going to have to be changed to be a classmethod. The problem
    is that if images are aligned or something, then they would actually end up having different
    dimensions. Which means that each Segment operation class could have it's own. One way to handle
    it would be to simply not save the model at all and leave it after Segment is done, but this seems
    like it could be inefficient if no aligning is done. Another option would be to save the models
    as a class attribute that are indexed by the dimension of the model. This would be more efficient
    but I'm worried about how much memory that's going to use. '''
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
                    input_layer = Concatenate(axis=3)([input_layer, trans_layers[s]])

                input_layer = _add_conv_module(input_layer, int(filt))

                # Fewer output features as upsampling progresses
                filt *= 0.5

            return input_layer

        def _add_conv_module(input_layer, filt: int = 64):
            """
            Build successive Conv-BatchNorm layers
            """
            for _ in range(layers):
                input_layer = conv_layer(filters=filt, kernel_size=kernel,
                                         strides=strides, activation=activation,
                                         padding=padding)(input_layer)
                input_layer = BatchNormalization(axis=-1, momentum=momentum)(input_layer)

            return input_layer

        # TODO: Will this break other models?
        K.clear_session()

        # Input
        x = Input(dimensions)
        # Pooling
        y, trans_layers = _add_pool_module(x)
        # Base layer
        new_filt = self.model_kws['init_filters'] * 2 ** len(trans_layers)
        y = _add_conv_module(y, filt=new_filt)
        # Upsampling
        filt = y.shape[-1] / 2
        y = _add_up_module(y, filt, trans_layers)
        # Output
        y = conv_layer(classes, 1)(y)
        y = Activation('softmax')(y)

        return tensorflow.keras.models.Model(x, y)
