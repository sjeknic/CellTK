from typing import Tuple, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, UpSampling1D,
                                     Conv2D, MaxPooling2D, UpSampling2D,
                                     BatchNormalization, Input, Activation,
                                     Concatenate)
from tensorflow.keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import tensorflow.keras.models

from cellst.utils._types import Image


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
                image: Image = None,
                roi: int = 2
                ) -> Image:
        """
        Use the given model to generate a mask of predictions
        """
        # First, image has to be normalized while it is still 3D
        image = self.normalize_image(image)

        # This is the expected dimension, add axis at -1 for channels
        if image.ndim == 3:
            image = np.expand_dims(image, -1)
        else:
            raise ValueError(f'Expected 3D stack, got {image.ndim}')

        # Pad the image so that it matches the model dimensions
        image, pads = self.pad_2d(image, self.model_kws['steps'])

        # TODO: Can these models be saved and called?
        # Predict the results and save in last axis
        out = self.model.predict(image)
        out = self.undo_padding(out, pads)
        out = self.normalize_result(out)

        return out[..., roi]

    def pad_model_dimension(self,
                            shape: Tuple[int],
                            model_steps: int
                            ) -> Tuple[int]:
        """
        Pads a shape to be divisible by the model
        """
        # Dimensions are independent of frames. height x width x channels(1)
        pads = self._calculate_pads(shape[:-1], 2 ** model_steps)
        # Remember to account for the channels dimension
        pads.append((0, 0))
        # Pads is a list of tuples that sums to the delta on each axis
        return tuple([int(s + sum(p)) for s, p in zip(shape, pads)])

    def pad_2d(self,
               image: Image,
               model_steps: int,
               mode: str = 'constant',
               constant_values: float = 0.
               ) -> Image:
        """
        Pads an image to be divisible by the model
        This assumes that image is supposed to be 4 dimensional.
        Batch x Height x Width x Channels
        """
        pads = self._calculate_pads(image.shape, 2 ** model_steps)
        # The channel and batch axes will not be padded
        pads[0] = (0, 0)
        pads[-1] = (0, 0)

        return np.pad(image, pads, mode=mode, constant_values=constant_values), pads

    def undo_padding(self,
                     image: Image,
                     pads: List[Tuple]
                     ) -> Image:
        """
        Trim image back down to the original size
        """
        pads = [slice(p[0], -p[1]) if (p[0] != 0) and (p[1] != 0) else slice(None)
                for p in pads]
        return image[tuple(pads)]

    def normalize_image(self,
                        image: Image,
                        low_percentile: float = 0.1,
                        high_percentile: float = 99.9
                        ) -> Image:
        """
        Normalizes the image to [0, 1]
        Percentiles prevent high/low outliers from impacting fit

        NOTE: The specific percentages are based on what was used during training
              with CellUNet. Results will vary if the percentiles are not the same.
        """
        # Find outlier values and clip
        low_perc = np.percentile(image, low_percentile)
        high_perc = np.percentile(image, high_percentile)
        clipped = np.clip(image, a_min=low_perc, a_max=high_perc)

        # Fit scaler on the clipped array. Cast it to N samples x 1 feature
        scaler = MinMaxScaler(clip=True).fit(clipped.reshape(-1, 1))

        # Transform the original image and reshape back to original shape
        return scaler.transform(image.reshape(-1, 1)).reshape(image.shape)

    def normalize_result(self, image: Image) -> Image:
        """
        Given an image ensures image.sum(-1) = 1 at each spot
        NaNs after division are cast to 0
        """
        for i in range(image.shape[0]):
            image[i, ...] = image[i, ...] / image[i, ...].sum(-1)[..., None]
        image[np.isnan(image)] = 0.

        return image

    def _calculate_pads(self, shape: Tuple[int], target_mod: int) -> List[Tuple]:
        """
        Calculate the adjustments to each axes in shape that will make it evenly
        divisible by the target_mod.
        """
        axes_deltas = [target_mod - (s % target_mod) if s % target_mod
                       else 0 for s in shape]

        if not all(a == 0 for a in axes_deltas):
            # Divide by two because padding both sides
            pads = [int(a / 2) for a in axes_deltas]

            # Adjust for odds
            pads = [(p, p + 1) if a % 2 else (p, p)
                    for (a, p) in zip(axes_deltas, pads)]
        else:
            pads = [(0, 0) * len(axes_deltas)]

        return pads

    @staticmethod
    def _build_model(dimensions: Tuple[int],
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
        new_filt = init_filters * 2 ** len(trans_layers)
        y = _add_conv_module(y, filt=new_filt)
        # Upsampling
        filt = y.shape[-1] / 2
        y = _add_up_module(y, filt, trans_layers)
        # Output - last layer is Conv 1x1
        # TODO: Add other activation options here
        y = conv_layer(filters=classes, kernel_size=1, strides=1,
                       activation=activation, padding=padding)(y)

        return tensorflow.keras.models.Model(x, y)
