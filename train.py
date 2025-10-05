import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

DATA_PATH = "data/iris.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open(os.path.join(MODEL_DIR, "iris_model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save metrics
accuracy = model.score(X_test, y_test)
pd.DataFrame({"accuracy": [accuracy]}).to_csv("metrics.csv", index=False)
print("Training complete, accuracy:", accuracy)
