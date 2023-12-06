import json
import pickle

import numpy as np
from typing import Optional

from multiple_linear_regression import MultipleLinearRegression


class ModelSaver:
    """
    The ModelSaver class is responsible for saving and loading the
    coefficients of a Multiple Linear Regression model to and from a file.
    It supports multiple file formats, including JSON and pickle.

    Methods:
        save: Saves the model coefficients to a specified
              file in a given format.
        load: Loads model coefficients from a specified file in a given format.
    """

    def save(
        self,
        model: 'MultipleLinearRegression',
        filename: str,
        file_format: str
    ) -> None:
        """
        Saves the coefficients of the provided MultipleLinearRegression model
        to a file.

        Args:
            model (MultipleLinearRegression): The regression model whose
                                              coefficients are to be saved.
            filename (str): The name of the file where the coefficients will
                            be saved.
            file_format (str): The format in which to save the model.
                               Supported formats: 'json', 'pickle'.

        Returns:
            None
        """
        if file_format == 'json':
            with open(filename, 'w') as f:
                json.dump(model.coefficients.tolist(), f)
        elif file_format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(model.coefficients, f)
        else:
            raise ValueError(
                "Unsupported file format. Please use 'json' or 'pickle'."
            )

    def load(self, filename: str, file_format: str) -> Optional[np.ndarray]:
        """
        Loads model coefficients from a specified file.

        Args:
            filename (str): The name of the file from which to load the
                            coefficients.
            file_format (str): The format of the file to load. Supported
                               formats: 'json', 'pickle'.

        Returns:
            Optional[np.ndarray]: The loaded coefficients as a numpy array, or
                                  None if the file could not be loaded.
        """
        if file_format == 'json':
            with open(filename, 'r') as f:
                coefficients = json.load(f)
            return np.array(coefficients)
        elif file_format == 'pickle':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(
                "Unsupported file format. Please use 'json' or 'pickle'."
            )
