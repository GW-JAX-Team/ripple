from abc import ABC, abstractmethod

from jaxtyping import Array, Float, PyTree


class WaveformModel(ABC):

    model_parameters: PyTree
    
    def __init__(self) -> None:
        pass

    def __call__(self, sample_points: Float[Array, " n_sample"], source_parameters: Float[Array, " n_params"], config_parameters: PyTree) -> Float[Array, "2 n_sample"]:
        """ Wrapper function on top of self.full_model using default model_parameters.
        
        Args:
            sample_points (Float[Array, " n_sample"]): The sample points at which the model is evaluated. This should be either in time or frequency.
            source_parameters (Float[Array, " n_params"]): The source parameters that define the model. This should include parameters such as the mass of the source.
            config_parameters (PyTree): Configuration parameters for the model. This includes parameters such as the reference frequency.
        
        Returns:
            Float[Array, "2 n_sample"]: The model output, typically a complex array representing the waveform in frequency or time domain. It assumes the first part of the array being the plus polarization and the second part being the cross polarization.    
        """
        return self.full_model(sample_points, source_parameters, config_parameters, self.model_parameters) 
        
    @abstractmethod
    def full_model(self, sample_points: Float[Array, " n_sample"],      source_parameters: Float[Array, " n_params"], config_parameters: PyTree, model_parameters: PyTree) -> Float[Array, "2 n_sample"]:
        """ The full model definition. This includes the model parameters such that one can leverage jax transfomation over that axis as well.
                
        Args:
            sample_points (Float[Array, " n_sample"]): The sample points at which the model is evaluated. It should be either in time or frequency same
            source_parameters (Float[Array, " n_params"]): The source parameters that define the model. This should include parameters such as the mass of the source.
            config_parameters (PyTree): Configuration parameters for the model. This includes parameters such as the reference frequency.
            model_parameters (PyTree): Model parameters that can be optimized or transformed. This should be parameters unique to this model.

        Returns:
            Float[Array, "2 n_sample"]: The model output, typically a complex array representing the waveform in frequency or time domain. It assumes the first part of the array being the plus polarization and the second part being the cross polarization.
        """
        raise NotImplementedError