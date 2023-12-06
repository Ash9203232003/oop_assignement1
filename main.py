import pandas as pd

from multiple_linear_regression import MultipleLinearRegression
from regression_plotter import RegressionPlotter
from model_saver import ModelSaver


def main():
    data = pd.read_csv('housing.csv')
    num_features = int(
        input("Enter the number of features to use (1, 2, or 5): ")
    )

    if num_features == 1:
        selected_columns = ['median_income']
    elif num_features == 2:
        selected_columns = ['latitude', 'longitude']
    elif num_features == 5:
        selected_columns = ['median_income', 'housing_median_age',
                            'total_rooms', 'population', 'latitude']
    else:
        print("Invalid number of features. Please select 1, 2, or 5.")
        return

    X = data[selected_columns].values
    y = data['median_house_value'].values

    model = MultipleLinearRegression()
    model.train(X, y)

    plotter = RegressionPlotter(model, feature_names=selected_columns,
                                target_name='Median House Value')
    plotter.plot(X, y)

    saver = ModelSaver()
    saver.save(model, 'model.json', 'json')
    loaded_coefficients = saver.load('model.json', 'json')
    print("Loaded coefficients:", loaded_coefficients)


if __name__ == "__main__":
    main()
