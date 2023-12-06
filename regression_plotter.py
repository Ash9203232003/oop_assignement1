import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

from multiple_linear_regression import MultipleLinearRegression


class RegressionPlotter:
    """
    This class is responsible for plotting regression lines or planes based on
    the trained Multiple Linear Regression model.

    Attributes:
        model (MultipleLinearRegression): The regression model whose results
                                          are to be plotted.
        feature_names (Optional[List[str]]): A list of names for the features
                                             used in the model. Defaults to
                                             None.
        target_name (str): The name of the target variable. Defaults to
                           'Target'.
    """

    def __init__(
        self,
        model: MultipleLinearRegression,
        feature_names: Optional[List[str]] = None,
        target_name: str = 'Target'
    ):
        """
        Initializes the RegressionPlotter with a regression model, optional
        feature names, and a target name.

        Args:
            model (MultipleLinearRegression): The regression model to plot.
            feature_names (Optional[List[str]]): Names of the features used in
                                                 the model. Defaults to None.
            target_name (str): The name of the target variable. Defaults to
                               'Target'.
        """
        self.model = model
        self.feature_names = feature_names
        self.target_name = target_name

    def plot(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plots the regression model results. Chooses the plot type based on the
        number of features.

        Args:
            X (np.ndarray): The input features used for prediction.
            y (np.ndarray): The actual target values.

        Returns:
            None
        """
        num_features = X.shape[1]

        if num_features == 1:
            self._plot_2d(X, y)
        elif num_features == 2:
            self._plot_3d(X, y)
        else:
            self._plot_multi_2d(X, y)

    def _plot_2d(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plots a 2D regression line when there is only one feature.

        Args:
            X (np.ndarray): The input feature used for prediction.
            y (np.ndarray): The actual target values.

        Returns:
            None
        """
        plt.scatter(X, y, color='blue')
        plt.plot(X, self.model.predict(X), color='red')
        plt.xlabel(self.feature_names[0] if self.feature_names else 'Feature')
        plt.ylabel(self.target_name)
        title = (f'Linear Regression with {self.feature_names[0]}'
                 if self.feature_names
                 else 'Linear Regression with One Feature')
        plt.title(title)
        plt.show()

    def _plot_3d(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plots a 3D regression plane when there are two features.

        Args:
            X (np.ndarray): The input features used for prediction.
            y (np.ndarray): The actual target values.

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, color='blue')

        # Creating a meshgrid for plotting the regression plane
        x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
        x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
        x0, x1 = np.meshgrid(x0_range, x1_range)
        x_plot = np.c_[x0.ravel(), x1.ravel()]
        y_plot = self.model.predict(x_plot).reshape(x0.shape)

        ax.plot_surface(x0, x1, y_plot, alpha=0.3, color='red')
        ax.set_xlabel(
            self.feature_names[0]
            if self.feature_names
            else 'Feature 1'
        )
        ax.set_ylabel(
            self.feature_names[1]
            if self.feature_names
            else 'Feature 2'
        )
        ax.set_zlabel(self.target_name)
        plt.title('Linear Regression with Two Features')
        plt.show()

    def _plot_multi_2d(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plots multiple 2D regression lines in separate windows when there are
        more than two features, setting non-considered attributes to their
        average values.

        Args:
            X (np.ndarray): The input features used for prediction.
            y (np.ndarray): The actual target values.

        Returns:
            None
        """
        averages = np.mean(X, axis=0)

        plt.ion()

        for i in range(X.shape[1]):
            plt.figure(i + 1)
            plt.scatter(X[:, i], y, color='blue')

            X_temp = np.tile(averages, (X.shape[0], 1))
            X_temp[:, i] = X[:, i]

            plt.plot(X[:, i], self.model.predict(X_temp), color='red')
            xlabel = (
                self.feature_names[i]
                if self.feature_names else
                f'Feature {i + 1}')
            plt.xlabel(xlabel)
            plt.ylabel(self.target_name)
            title = (
                f'Linear Regression ({xlabel} vs {self.target_name})'
                if self.feature_names
                else f'Linear Regression (Feature {i + 1} vs Target)')
            plt.title(title)

            plt.tight_layout()
            plt.show(block=True)  # Set block to True to wait for close
