import numpy as np
from typing import Optional


class MultipleLinearRegression:
    """
    This class implements a Multiple Linear Regression model.

    Attributes:
        coefficients (Optional[np.ndarray]): Coefficients of the regression
                                             model. Initialized as None.
    """

    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Multiple Linear Regression model using the normal equation.

        Args:
            X (np.ndarray): A 2D numpy array of input features.
            y (np.ndarray): A 1D numpy array of target values.

        Returns:
            None
        """
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        try:
            self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        except np.linalg.LinAlgError:
            print("Cannot invert matrix: the matrix is singular.")
            self.coefficients = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Linear Regression model.

        Args:
            X (np.ndarray): A 2D numpy array of input features.

        Returns:
            np.ndarray: Predicted values.
        """
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_b @ self.coefficients
