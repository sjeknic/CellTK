import functools
import warnings
from typing import Tuple, List, Collection

import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import skimage.filters as filt
import skimage.feature as feat
import skimage.util as util
import sklearn.preprocessing as preproc
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, UpSampling1D,
                                     Conv2D, MaxPooling2D, UpSampling2D,
                                     BatchNormalization, Input, Activation,
                                     Concatenate, Layer)
from tensorflow.keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import tensorflow.keras.models
import celltk.external.misic.extras as miext
import celltk.external.misic.utils as miutil

from celltk.utils._types import Image
from celltk.utils.operation_utils import nan_helper_2d, get_split_idxs


# TODO: Write docstring explaining kwargs
# TODO: Add tf.saved_model.LoadOptions(allow_partial_checkpoint=True)
class _UNetStructure():
    _model_kws = {}
    _alpha = 0.3

    def __init__(self,
                 dimensions: Tuple[int],
                 weight_path: str,
                 ) -> None:
        # Build model
        adj_dim = self.pad_model_dimension(dimensions, self._model_kws['steps'])
        input_layer, output_layer = self._build_model(adj_dim, **self._model_kws)
        self.model = self._compile_model(input_layer, output_layer)

        # Make model not trainable - prevents some problems with loading weights
        for n, l in enumerate(self.model.layers):
            self.model.layers[n].trainable = False

        self.model.load_weights(weight_path).expect_partial()

    @classmethod
    def _build_model(cls,
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
            activation = lambda x: LeakyReLU(alpha=cls._alpha)(x)

        # TODO: Any args that change need to be passed around
        def _add_pool_module(input_layer: Layer,
                             init_filters: int):
            trans_layers = []
            filt = init_filters
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

        def _add_up_module(input_layer: Layer,
                           init_filters: int,
                           trans_layers: list = []):
            # trans_layers starts at the top of the model,
            # we are the bottom when this module starts
            trans_layers = trans_layers[::-1]

            filt = init_filters
            for s in range(steps):
                input_layer = up_layer()(input_layer)

                if transfer:
                    # TODO: Axis cannot be hard-coded as 3
                    input_layer = Concatenate(axis=-1)([input_layer, trans_layers[s]])

                input_layer = _add_conv_module(input_layer, int(filt))

                # Fewer output features as upsampling progresses
                filt *= 0.5

            return input_layer

        def _add_conv_module(input_layer: Layer, filt: int):
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
        y, trans_layers = _add_pool_module(x, init_filters)
        # Base layer
        new_filt = init_filters * 2 ** steps
        y = _add_conv_module(y, filt=new_filt)
        # Upsampling
        filt = y.shape[-1] / 2
        y = _add_up_module(y, filt, trans_layers)
        # Output - last layer is Conv 1x1
        # TODO: Add other activation options here
        y = conv_layer(filters=classes, kernel_size=1, strides=1,
                       activation=activation, padding=padding)(y)

        return x, y

    @classmethod
    def _compile_model(cls,
                       input_layer: Layer,
                       output_layer: Layer
                       ) -> tensorflow.keras.models.Model:
        """"""
        return tensorflow.keras.models.Model(input_layer, output_layer)

    def _calculate_pads(self,
                        shape: Tuple[int],
                        target_mod: int,
                        bidirectional: bool = True
                        ) -> List[Tuple]:
        """
        Calculate the adjustments to each axes in shape that will make it evenly
        divisible by the target_mod.
        """
        axes_deltas = [target_mod - (s % target_mod) if s % target_mod
                       else 0 for s in shape]

        if not all(a == 0 for a in axes_deltas):
            if bidirectional:
                # Divide by two because padding both sides
                pads = [int(a / 2) for a in axes_deltas]

                # Adjust for odds
                pads = [(p, p + 1) if a % 2 else (p, p)
                        for (a, p) in zip(axes_deltas, pads)]
            else:
                pads = [(0, a) for a in axes_deltas]
        else:
            pads = [(0, 0) * len(axes_deltas)]

        return pads

    def _weighted_categorical_crossentropy(self, weights: np.ndarray) -> float:
        """ A weighted version of keras.objectives.categorical_crossentropy

        Args:
            weights: numpy array of shape (C,) where C is the number of classes

        Returns:
            Weighted categorical crossentropy as float
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so class probs of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss

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


class FluorUNetModel(_UNetStructure):
    """
    TODO:
        - Add option to save and load models from JSON
        - Add option to pass custom models
        - Add custom load weights function
    """
    _model_kws = dict(classes=3, kernel=(3, 3), steps=3, layers=2,
                      init_filters=64, transfer=True, activation='relu',
                      padding='same', mode=0, momentum=0.9)

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
        image, pads = self.pad_2d(image, self._model_kws['steps'])

        # TODO: Can these models be saved and called?
        # Predict the results and save in last axis
        out = self.model.predict(image)
        out = self.undo_padding(out, pads)
        out = self.normalize_result(out)

        return out[..., roi]

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

        return (np.pad(image, pads, mode=mode,
                       constant_values=constant_values),
                pads)

    def undo_padding(self,
                     image: Image,
                     pads: List[Tuple]
                     ) -> Image:
        """
        Trim image back down to the original size
        """
        slices = [slice(None)] * len(pads)
        for n, p in enumerate(pads):
            st = p[0] if p[0] else 0
            en = -1 * p[1] if p[1] else 0
            if st or en: slices[n] = slice(st, en)

        return image[tuple(slices)]

    def normalize_image(self,
                        image: Image,
                        low_percentile: float = 0.1,
                        high_percentile: float = 99.9
                        ) -> Image:
        """Normalizes input image to [0, 1]

        Uses image percentile values to prevent high/low outliers
        from impacting fit.

        Args:

        Returns:

        NOTE: If training wasn't done with this normalization, it
        might affect the results.
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


class OrigUNetModel(FluorUNetModel):
    _model_kws = dict(classes=3, kernel=(3, 3), steps=4, layers=2,
                      init_filters=64, transfer=True, activation='relu',
                      padding='same', mode=0, momentum=0.9)


class UPeakModel(_UNetStructure):
    """
    Regions of interest:
    0 - Background
    1 - Slope
    """
    _norm_methods = ['amplitude', 'zscore']
    _norm_kwargs = [{}, {}]
    _norm_inplace = [True, False]
    _model_kws = dict(classes=2, kernel=4, steps=2, layers=2,
                      init_filters=64, transfer=False, activation='leaky',
                      padding='same', mode=0, momentum=0.9)

    def __init__(self,
                 weight_path: str,
                 dimensions: Tuple[int] = None
                 ) -> None:
        """"""
        if dimensions:
            # If dimensions are known, make model
            super().__init__(dimensions, weight_path)
        else:
            # Otherwise, just save the weight path
            self.weight_path = weight_path

    @classmethod
    def _compile_model(cls,
                       input_layer: Layer,
                       output_layer: Layer
                       ) -> tensorflow.keras.models.Model:
        """"""
        # An activation layer needs to be added for UPeak
        output_layer = Activation('softmax')(output_layer)
        return tensorflow.keras.models.Model(input_layer, output_layer)

    def predict(self,
                array: np.ndarray,
                roi: Tuple[int] = 1,
                min_nonnan: int = 2
                ) -> np.ndarray:
        """
        By default, returns slope and plateau summed
        """
        # Create default output - all nans
        out = np.empty(array.shape)
        out[:] = np.nan

        # Cells with all or nearly all nans need to be removed
        # Mask is saved, so that they can be added back after it's all done
        nan_mask = np.isnan(array).sum(1) > min_nonnan
        nonan_arr = array[~nan_mask, :]
        # check that something is left...
        if nonan_arr.shape[0] == 0: return out
        if nonan_arr.ndim == 1: nonan_arr = nonan_arr[:, None]

        # Normalize the inputs
        nonan_arr = self.normalize_inputs(nonan_arr)
        nonan_arr, pads = self.pad_1d(nonan_arr, self._model_kws['steps'])

        # Check if a model needs to be built
        if not hasattr(self, 'model'):
            super().__init__(nonan_arr.shape, self.weight_path)

        # Predict - batch should not be required
        preds = self.model.predict(nonan_arr)
        preds = self.undo_padding(preds, pads)

        # Add predictions back to out
        out[~nan_mask, :] = preds[..., roi]

        # Returns a 3D arr...
        return out

    def set_normalization_method(self,
                                 method: Collection[str] = None,
                                 kwargs: Collection[dict] = None,
                                 inplace: Collection[bool] = None
                                 ) -> None:
        """Not sure about how to handle this, but want to pass inputs
        without having to first call normalization. So thinking, set method
        then call predict on the images, and they will be normalized first

        pass nothing to skip all normalization
        """
        # Save inputs
        self._norm_methods = method if method else []
        self._norm_kwargs = kwargs if kwargs else []
        self._norm_inplace = inplace if inplace else []

        # Check that all are the same length
        lens = [len(l) for l in [self._norm_methods,
                                 self._norm_kwargs,
                                 self._norm_inplace]]
        assert len(set(lens)) == 3, f'Lengths of inputs must match. Got {lens}'

    def normalize_inputs(self,
                         array: np.ndarray
                         ) -> np.ndarray:
        """"""
        def _dim_handler(func):
            # Make decorator for handling output dimensions
            @functools.wraps(func)
            def wrapper(arr, kws, inplace):
                _res = func(arr, **kws)

                # Inp is true if _res should replace array
                if inplace:
                    pass
                # Otherwise, save it as an additional feature
                else:
                    # Need to check dims first - always return 3D
                    if arr.ndim == 2: arr = arr[..., np.newaxis]
                    if _res.ndim == 2: _res = _res[..., np.newaxis]

                    _res = np.concatenate((arr, _res), axis=-1)

                return _res
            return wrapper

        # Linearly interpolate any nans if they exist
        if np.isnan(array).any():
            array = nan_helper_2d(array)

        # Iterate through the defined inputs
        inputs = zip(self._norm_methods,
                     self._norm_kwargs,
                     self._norm_inplace)
        for m, k, i in inputs:
            if m == 'amplitude':
                array = _dim_handler(self.normalize_amplitude)(array, k, i)
            elif m == 'zscore':
                array = _dim_handler(self.normalize_zscore)(array, k, i)
            elif m == 'cwt':
                array = _dim_handler(self.create_cwt)(array, k, i)
            else:
                warnings.warn(f'Did not understand norm method {m}')

        return array

    def pad_model_dimension(self,
                            shape: Tuple[int],
                            model_steps: int
                            ) -> Tuple[int]:
        """
        Pads a shape to be divisible by the model

        For UPeak, this will likely get a 3D array (instead of 2D)
        Skip the first axis, which is just the number of traces
        """
        if len(shape) == 3:
            dims = shape[1:]
        elif len(shape) == 2:
            dims = shape
        else:
            raise ValueError(f'Invalid input shape: {shape}')

        return super().pad_model_dimension(dims, self._model_kws['steps'])

    def pad_1d(self,
               array: np.ndarray,
               model_steps: int,
               mode: str = 'edge',
               constant_values: float = 0
               ) -> np.ndarray:
        '''
        pad_mode is edge or constant.
        if edge, repeats last value from trace. if constant, pads with cv
        array are padded at the end
        '''
        if mode == 'constant':
            options_dict = {'constant_values': constant_values}
        else:
            options_dict = {}

        pads = self._calculate_pads(array.shape, 2 ** model_steps,
                                    bidirectional=False)

        # Only the trace length needs to be padded
        pads[0] = (0, 0)
        pads[-1] = (0, 0)

        return np.pad(array, pad_width=pads, mode=mode, **options_dict), pads

    def undo_padding(self,
                     array: Image,
                     pads: List[Tuple]
                     ) -> Image:
        """Trim traces back down to the original size

        Can assume that the array was only padded at the end
        """
        pads = [slice(0, -p[1]) if p[1] else slice(None) for p in pads]
        return array[tuple(pads)]

    def normalize_amplitude(self,
                             array: np.ndarray,
                             by_row: bool = True
                             ) -> np.ndarray:
        """This function should not be directly called, but will
        get called by normalize_inputs. Not private for API documentation

        TODO: Test applied to correct axis
        TODO: by_row did not work, so I removed it
        """
        assert array.ndim == 2  # Normalize before adding features
        input_shape = array.shape
        array = preproc.maxabs_scale(array, axis=1)[..., None]
        return array

    def normalize_zscore(self,
                          array: np.ndarray,
                          by_row: bool = True,
                          offset: float = 0.,
                          norm: bool = False
                          ) -> np.ndarray:
        """TODO: Test applied to correct axis
        I don't think this works. Normalizes all features independently??
        """
        assert np.squeeze(array).ndim == 2  # Normalize before adding features
        input_shape = array.shape
        # Move all samples into same row
        if not by_row:
            array = array.copy().reshape(1, -1)

        # z-score function including the given offset, apply to each row
        array = stats.zscore(array, axis=1, nan_policy='omit') + offset

        # Normalize to [-1, 1] if needed
        if norm:
            array = MaxAbsScaler().fit_transform(array)

        return array.reshape(input_shape)

    def create_cwt(self,
                   traces: np.ndarray,
                   widths: np.ndarray = None,
                   wavelet = signal.ricker
                   ) -> np.ndarray:
        """"""
        # Get sizes of the wavelets
        splits = np.array_split(traces, 64, axis=1)
        widths = get_split_idxs(splits, axis=1)

        out = np.zeros((traces.shape[0], traces.shape[1], len(widths)), dtype=float)
        for idx, tr in enumerate(traces):
            tr = np.squeeze(tr)
            assert tr.ndim == 1
            if np.isnan(tr).any():
                # Assume traces were interpolated and then padded with nans
                # Need to remove any trailing nans - More for UPeak than here...
                first_nan = np.logical_not(np.isnan(tr)).argmin()
                tr = tr[:first_nan]
            else:
                first_nan = None

            cwt = signal.cwt(tr, wavelet=wavelet, widths=widths).T

            out[idx, :first_nan, : ] = cwt

        return out


class MisicModel(FluorUNetModel):
    # TODO: Add logging of citation information
    # TODO: Add custom build of structure
    '''
    NOTE: I tried building out a model with a separate input layer that
    would allow for calculations on the whole frame at once. Unfortunately,
    I couldn't get this to work. Possibly completely unrelated to the model
    structure though, so if I have time, then maybe something to try. I'm leaving
    the reconfigure function I wrote in here.
    '''
    _model_kws = {'steps': 4, 'channels': 3}

    def __init__(self,
                 model_path: str,
                 ) -> None:
        self.model = tensorflow.keras.models.load_model(model_path)

    def misic_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Uses recommended preprocessing from MiSiC model

        Returns
            np.ndarray: (fr, h, w)
        """
        out = np.zeros_like(image).astype(np.float64)
        # Image is #D here (frames, h, w)
        for fr, im in enumerate(image):
            im = miext.adjust_gamma(im, 0.25)
            im = filt.unsharp_mask(im, 2.2, 0.6)
            # NOTE: nvert is false for dark backgrounds?
            im = miext.add_noise(im, sensitivity=0.13, invert=True)
            out[fr, ...] = im

        return out

    def misic_postprocess(self,
                          original: np.ndarray,
                          tiles: np.ndarray,
                          params: List[dict],
                          n_tiles: int
                          ) -> np.ndarray:
        """"""
        # Reconstruct the prediictions from the tiles
        preds = np.split(tiles, tiles.shape[0] // n_tiles, axis=0)
        preds = [miutil.stitch_tiles(i, params) for i in preds]
        predictions = np.squeeze(np.stack(preds, axis=0))

        # Apply the predictions with watershed to original image
        out = np.zeros_like(original)
        for fr, (im, pred) in enumerate(zip(original, predictions)):
            out[fr, ...] = miext.postprocessing(im, pred)

        return out

    def normalize_inputs(self, tiles: List[np.ndarray]):
        """Normalizes intensity and adjusts dimensions of channel axis"""
        return np.array([self.shapeindex_preprocess(
                                miutil.normalize2max(np.squeeze(t))
                         )
                         for ti in tiles for t in ti])

    def predict(self, image: np.ndarray) ->  np.ndarray:
        """"""
        # TODO: Add rescaling for different size bacteria?
        pre_image = self.misic_preprocess(image)

        # Split the image up into tiles as MiSiC expects
        all_tiles = []
        for im in pre_image:
            tiles, params = miutil.extract_tiles(im, size=256, exclude=16)
            # Because all images are same size, n_tiles and params should be same
            n_tiles = tiles.shape[0]
            all_tiles.append(tiles)

        # Multiply by -1 because using light background
        tiles = -1 * self.normalize_inputs(all_tiles)
        tiles = np.stack(tiles, axis=0)

        # Predict the results
        out = self.model.predict(tiles)
        out = self.misic_postprocess(image, out, params, n_tiles)

        return util.img_as_uint(out)

    def _reconfigure_input_layer(self,
                                 model: tensorflow.keras.models.Model,
                                 dimensions: Tuple[int]
                                 ) -> tensorflow.keras.models.Model:
        """MiSiC loads directly from the file. Need to change input shape."""
        # Get the initial configuration
        old_model = model
        model_config = model.get_config()

        # Reconfigure input dimensions
        model_config['layers'][0]['name'] = 'new_input'
        if len(dimensions) == 3: dimensions = (None, *dimensions)
        model_config['layers'][0]['config']['batch_input_shape'] = dimensions
        model_config['layers'][0]['config']['name'] = 'new_input'

        # Reconfigure first convolutional layer
        model_config['layers'][1]['inbound_nodes'] = [[['new_input', 0, 0, {}]]]
        model_config['input_layers'] = [['new_input', 0, 0]]

        # Rebuild model
        new_model = tensorflow.keras.models.Model.from_config(model_config)

        # Re-apply weights - skip input
        for new, old in zip(new_model.layers[1:], old_model.layers[1:]):
            new.set_weights(old.get_weights())

        return new_model

    def shapeindex_preprocess(self, im):
        """Copied directly from MiSiC (sjeknic)"""
        ''' apply shap index map at three scales'''
        sh = np.zeros((im.shape[0],im.shape[1],3))
        if np.max(im) ==0:
            return sh
        # pad to minimize edge artifacts
        sh[:,:,0] = feat.shape_index(im,1, mode='reflect')
        sh[:,:,1] = feat.shape_index(im,1.5, mode='reflect')
        sh[:,:,2] = feat.shape_index(im,2, mode='reflect')
        return sh
