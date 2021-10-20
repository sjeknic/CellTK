from typing import Collection, Tuple

from cellst.operation import Operation
from cellst.utils._types import Image


class Process(Operation):
    _input_type = (Image,)
    _output_type = Image

    def __init__(self,
                 input_images: Collection[str] = ['channel000'],
                 output: str = 'process',
                 save: bool = False,
                 _output_id: Tuple[str] = None,
                 ) -> None:
        super().__init__(output, save, _output_id)

        if isinstance(input_images, str):
            self.input_images = [input_images]
        else:
            self.input_images = input_images

        self.output = output
