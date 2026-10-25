import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("mobile.csv")
features = ["battery_power", "ram", "px_height", "px_width", "int_memory"]
X, y = df[features], df["price_range"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte), zero_division=0))
values = [float(input(f"{c}: ")) for c in features]
print("Predicted price range class:", model.predict([values])[0])
