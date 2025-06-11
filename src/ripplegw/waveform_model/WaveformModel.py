from abc import ABC, abstractmethod
from jaxtyping import Float, Int, Array, PyTree

class WaveformModel(ABC):

    model_parameters: PyTree
    
    def __init__(self) -> None:
        pass

    def __call__(self, sample_points: Float[Array, " n_sample"], source_parameters: Float[Array, " n_params"]):
       return self.full_model(sample_points, source_parameters, self.model_parameters) 
        
    @abstractmethod
    def full_model(self, sample_points: Float[Array, " n_sample"],      source_parameters: Float[Array, " n_params"], model_parameters: PyTree):
        raise NotImplementedError