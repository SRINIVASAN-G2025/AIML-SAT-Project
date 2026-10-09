import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score, f1_score, precision_score

df = pd.read_csv("Cervical.csv")

y = pd.to_numeric(df["Biopsy"])

X = df.drop(columns=[
    "Biopsy", "Dx:Cancer", "Dx:CIN", "Dx:HPV",
    "Dx", "Hinselmann", "Schiller", "Citology"
])

X = X.replace("?", float("nan"))
X = X.apply(pd.to_numeric)

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X.columns
)
X_test = pd.DataFrame(
    imputer.transform(X_test), columns=X.columns
)

selector = SelectFromModel(
    RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight="balanced"
    ),

    
)

selector.fit(X_train, y_train)
X_train_B = selector.transform(X_train)
X_test_B = selector.transform(X_test)

models = {
    "Model A - All Features": (X_train, X_test),
    "Model B - Selected Features": (X_train_B, X_test_B)
}

results = {}
for name, (train_data, test_data) in models.items():

    start = time.time()

    model = GridSearchCV(
        RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        },
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    model.fit(train_data, y_train)
    predictions = model.predict(test_data)

    elapsed = time.time() - start

    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    results[name] = {
        "Recall": recall,
        "F1 Score": f1,
        "Precision": precision,
        "Time": elapsed,
        "Features": train_data.shape[1]
    }

    print("\n", name)
    print("Best Parameters:", model.best_params_)
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))
    print("Precision:", round(precision, 3))
    print("Execution Time:", round(elapsed, 3), "seconds")
    print("Number of Features:", train_data.shape[1])

importance_model = RandomForestClassifier(
    n_estimators=100, random_state=42,
    class_weight="balanced"
)
importance_model.fit(X_train, y_train)

importance = pd.Series(
    importance_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features" \
":")
print(importance.head(10))
print("\nFINAL PERFORMANCE COMPARISON")
print(pd.DataFrame(results).round(3).T)


