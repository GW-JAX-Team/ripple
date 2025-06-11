from abc import ABC
from jaxtyping import Float, Int, Array, PyTree

class WaveformModel(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, sample_points: Float[Array, " n_sample"], source_parameters: Float[Array, " n_params"], model_parameters: PyTree):
        pass

    def full_model(self, sample_axis:
        False
