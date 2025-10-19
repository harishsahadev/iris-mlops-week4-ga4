import pickle
import pandas as pd

def test_model_prediction():
    # Load trained model
    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Input sample with correct feature names
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )

    # Predict
    pred = model.predict(sample)

    # Check prediction is valid
    assert pred[0] in ["setosa", "versicolor", "virginica"], f"Unexpected prediction: {pred[0]}"
