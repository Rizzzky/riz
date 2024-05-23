import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = {
    'Kuis': [75, 60, 85, 70, 50, 90, 65, 80, 55, 85, 60, 75, 90, 70, 50, 65, 80, 55, 85, 60, 75, 50, 90, 65, 80, 55, 85, 60, 75, 90, 70, 50, 65, 80, 55, 85, 60, 75, 90, 65],
    'Tugas': [80, 65, 90, 75, 55, 85, 70, 85, 60, 90, 70, 80, 85, 75, 55, 70, 85, 60, 90, 70, 80, 55, 85, 70, 85, 60, 90, 70, 80, 85, 75, 55, 70, 85, 60, 90, 70, 80, 85, 70],
    'UTS': [70, 55, 80, 60, 40, 95, 50, 75, 45, 85, 50, 70, 95, 65, 40, 55, 80, 45, 85, 55, 70, 40, 95, 55, 75, 45, 85, 50, 70, 95, 65, 40, 55, 80, 45, 85, 55, 70, 95, 55],
    'UAS': [85, 70, 95, 80, 60, 90, 75, 85, 65, 95, 70, 85, 90, 80, 60, 75, 85, 65, 95, 70, 85, 60, 90, 75, 85, 65, 95, 70, 85, 90, 80, 60, 75, 85, 65, 95, 70, 85, 90, 75],
    'Target Lulus': ['Lulus', 'Tidak Lulus', 'Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Lulus', 'Lulus', 'Tidak Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Lulus', 'Lulus', 'Tidak Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Lulus', 'Tidak Lulus']
}

df = pd.DataFrame(data)

df['Target Lulus'] = df['Target Lulus'].map({'Lulus': 1, 'Tidak Lulus': 0})

X = df[['Kuis', 'Tugas', 'UTS', 'UAS']]
y = df['Target Lulus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
