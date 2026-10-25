import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv("mobile.csv")
features=["battery_power","ram","px_height","px_width","int_memory"]
x = df[features]
y = df["price_range"]
xtr, xte, ytr, yte= train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(xtr,ytr)
y_pred = model.predict(xte)
print("Classification Report:")
print(classification_report(yte,y_pred))
vals = [float(input(f"{col}: ")) for col in features]
prediction = model.predict([vals])[0]
print("Predicted price range class:",prediction)  