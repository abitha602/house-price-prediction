# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    # Load datasetxl
    try:
        df = pd.read_csv(r"C:\Users\ANAND\PycharmProjects\PythonProject5\data.csv")
        df = df.rename(columns={"price": "PRICE"})
        df = df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_data(df):
    # Split data
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# def train_model(X_train, y_train):
#     # Train model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     return model

def train_models(X_train, y_train):

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

# def evaluate_model(model, X_test, y_test):
#     # Evaluate model
#     score = model.score(X_test, y_test)
#     y_pred = model.predict(X_test)
#
#     print("Predicted Prices:")
#     print(y_pred[:10])  # show first 10 predictions
#     for actual, predicted in zip(y_test[:10], y_pred[:10]):
#         plt.bar(actual,predicted)
#         print(f"Actual: {actual}  |  Predicted: {predicted}")
#     return score

def evaluate_model(model, X_test, y_test):
    # score = model.score(X_test, y_test)
    # y_pred = model.predict(X_test)
    #
    # print("R-squared Score:", score)
    #
    # actual = y_test[:10]
    # predicted = y_pred[:10]
    #
    # for a, p in zip(actual, predicted):
    #     print(f"Actual: {a} | Predicted: {p}")
    #
    # x = np.arange(len(actual))
    #
    # plt.bar(x-0.2, actual, width=0.4, label="Actual")
    # plt.bar(x+0.2, predicted, width=0.4, label="Predicted")
    #
    # plt.xlabel("Test Samples")
    # plt.ylabel("Price")
    # plt.title("Actual vs Predicted Prices")
    # plt.legend()
    # plt.show()
    # return score, actual, predicted
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    print("R-squared Score:", score)

    # ---- Error Calculations (Added) ----
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    # -----------------------------------

    actual = y_test[:10]
    predicted = y_pred[:10]

    for a, p in zip(actual, predicted):
        print(f"Actual: {a} | Predicted: {p}")

    x = np.arange(len(actual))

    plt.bar(x - 0.2, actual, width=0.4, label="Actual")
    plt.bar(x + 0.2, predicted, width=0.4, label="Predicted")

    plt.xlabel("Test Samples")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.show()
    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals)

    plt.axhline(y=0, linestyle='--')

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")

    plt.show()

    return score, actual, predicted,mae, mse, rmse




def main():
    # Specify the file path
    file_path = r"C:\Users\ANAND\PycharmProjects\PythonProject5\data.csv"

    # Load data
    df = load_data(file_path)

    if df is not None:
        # Split data
        X_train, X_test, y_train, y_test = split_data(df)

        # Train model
        # model = train_model(X_train, y_train)
        #
        # # Evaluate model
        # score = evaluate_model(model, X_test, y_test)
        #
        # # Print score
        # print(f'R-squared Score: {score:.2f}')
        models = train_models(X_train, y_train)
        mae_results = {}
        mse_results = {}
        rmse_results = {}
        results = {}
        for name, model in models.items():
            print("\n==========================")
            print("Model:", name)

            score, actual, predicted,mae, mse, rmse = evaluate_model(model, X_test, y_test)

            results[name] = score
            mae_results[name] = mae
            mse_results[name] = mse
            rmse_results[name] = rmse

            # Print comparison
        print("\nModel Comparison")
        for model_name, score in results.items():
            print(f"{model_name} : {score}")

        # Plot comparison graph
        plt.bar(results.keys(), results.values())

        plt.xlabel("Models")
        plt.ylabel("R-Squared Score")
        plt.title("Model Performance Comparison")

        plt.show()
        # Error Comparison Graph
        models_list = list(mae_results.keys())
        x = np.arange(len(models_list))

        plt.bar(x - 0.2, list(mae_results.values()), width=0.2, label="MAE")
        plt.bar(x, list(mse_results.values()), width=0.2, label="MSE")
        plt.bar(x + 0.2, list(rmse_results.values()), width=0.2, label="RMSE")

        plt.xticks(x, models_list)

        plt.xlabel("Models")
        plt.ylabel("Error Values")
        plt.title("Error Comparison of Models")

        plt.legend()
        plt.show()
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        print("\nBest Model:", best_model_name)

        # Save model
        # joblib.dump((best_model, best_model_name), "house_price_prediction.pkl")
        joblib.dump((best_model, best_model_name, actual, predicted,score),
                    "house_price_prediction.pkl")
        print("Model saved as house_price_model.pkl")

if __name__ == "__main__":
    main()