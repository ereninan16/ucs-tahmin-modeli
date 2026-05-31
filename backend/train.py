"""
UCS Tahmin Modeli - Model Egitim Betigi
Kaynak: ucs_tahmin_modeli.ipynb (Eren INAN - Bitirme Calismasi II)
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

np.random.seed(42)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def granit_uret(n_numune=80):
    rho = np.random.normal(2.70, 0.05, n_numune)
    n = np.random.uniform(0.5, 3.0, n_numune)
    Vp = np.random.normal(5.5, 0.4, n_numune)
    Rn = np.random.normal(57, 4, n_numune)
    Is50 = np.random.uniform(5, 12, n_numune)
    UCS = (8 * Vp**1.25 + -6 * n + 1.2 * Rn + 5 * Is50 + 15 * (rho - 2.5))
    UCS = UCS + np.random.normal(0, 15, n_numune)
    UCS = np.clip(UCS, 100, 250)
    return pd.DataFrame({
        "Kaya_Tipi": "Granit",
        "Vp": np.round(Vp, 2), "n": np.round(n, 2),
        "rho": np.round(rho, 3), "Rn": np.round(Rn, 1),
        "Is50": np.round(Is50, 2), "UCS": np.round(UCS, 1),
    })


def kirectasi_uret(n_numune=90):
    rho = np.random.normal(2.55, 0.08, n_numune)
    n = np.random.uniform(2, 12, n_numune)
    Vp = np.random.normal(4.2, 0.5, n_numune)
    Rn = np.random.normal(45, 5, n_numune)
    Is50 = np.random.uniform(2, 7, n_numune)
    UCS = (7 * Vp**1.15 + -3 * n + 1.0 * Rn + 5 * Is50 + 12 * (rho - 2.3))
    UCS = UCS + np.random.normal(0, 10, n_numune)
    UCS = np.clip(UCS, 30, 150)
    return pd.DataFrame({
        "Kaya_Tipi": "Kirectasi",
        "Vp": np.round(Vp, 2), "n": np.round(n, 2),
        "rho": np.round(rho, 3), "Rn": np.round(Rn, 1),
        "Is50": np.round(Is50, 2), "UCS": np.round(UCS, 1),
    })


def kumtasi_uret(n_numune=80):
    rho = np.random.normal(2.35, 0.08, n_numune)
    n = np.random.uniform(8, 25, n_numune)
    Vp = np.random.normal(3.3, 0.4, n_numune)
    Rn = np.random.normal(35, 5, n_numune)
    Is50 = np.random.uniform(1, 5, n_numune)
    UCS = (6 * Vp**1.1 + -2 * n + 0.9 * Rn + 4 * Is50 + 10 * (rho - 2.1))
    UCS = UCS + np.random.normal(0, 7, n_numune)
    UCS = np.clip(UCS, 20, 100)
    return pd.DataFrame({
        "Kaya_Tipi": "Kumtasi",
        "Vp": np.round(Vp, 2), "n": np.round(n, 2),
        "rho": np.round(rho, 3), "Rn": np.round(Rn, 1),
        "Is50": np.round(Is50, 2), "UCS": np.round(UCS, 1),
    })


def train():
    print("Veri uretiliyor...")
    df = pd.concat(
        [granit_uret(80), kirectasi_uret(90), kumtasi_uret(80)],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    FEATURES = ["Vp", "n", "rho", "Rn", "Is50"]
    X = df[FEATURES]
    y = df["UCS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["Kaya_Tipi"]
    )

    print("XGBoost egitiliyor...")
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    print("Linear Regression egitiliyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    joblib.dump(xgb, os.path.join(MODEL_DIR, "model_xgboost.pkl"))
    joblib.dump(lr, os.path.join(MODEL_DIR, "model_linear_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

    from sklearn.metrics import r2_score
    r2_xgb = r2_score(y_test, xgb.predict(X_test))
    r2_lr = r2_score(y_test, lr.predict(scaler.transform(X_test)))
    print(f"XGBoost Test R2: {r2_xgb:.4f}")
    print(f"Linear Reg Test R2: {r2_lr:.4f}")
    print(f"Modeller kaydedildi: {MODEL_DIR}")


def train_v2():
    """v2: Gercek veriyle egitilmis 4-parametreli SVR modeli (yogunluk yok)."""
    import json
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ucs_gercek_veri_224.csv")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    y = df["UCS"].values

    print("v2a (tursuz SVR) egitiliyor...")
    Xa = df[["Vp_ms", "n", "SHR", "Is50"]].values
    model_v2a = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=500, gamma="auto", epsilon=0.1)),
    ])
    model_v2a.fit(Xa, y)
    joblib.dump(model_v2a, os.path.join(MODEL_DIR, "model_v2a.pkl"))

    print("v2b (kaya turlu SVR) egitiliyor...")
    Xb = df[["Vp_ms", "n", "SHR", "Is50", "RockType"]].copy()
    preproc = ColumnTransformer([
        ("num", StandardScaler(), ["Vp_ms", "n", "SHR", "Is50"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["RockType"]),
    ])
    model_v2b = Pipeline([
        ("prep", preproc),
        ("svr", SVR(kernel="rbf", C=500, gamma="auto", epsilon=0.1)),
    ])
    model_v2b.fit(Xb, y)
    joblib.dump(model_v2b, os.path.join(MODEL_DIR, "model_v2b.pkl"))

    kaya_turleri = sorted(df["RockType"].unique().tolist())
    with open(os.path.join(MODEL_DIR, "kaya_turleri.json"), "w", encoding="utf-8") as f:
        json.dump(kaya_turleri, f, ensure_ascii=False)

    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score, KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2a = cross_val_score(model_v2a, Xa, y, cv=kf, scoring="r2").mean()
    cv_r2b = cross_val_score(model_v2b, Xb, y, cv=kf, scoring="r2").mean()
    print(f"v2a CV R2: {cv_r2a:.4f}")
    print(f"v2b CV R2: {cv_r2b:.4f}")
    print(f"Kaya turleri: {kaya_turleri}")
    print("v2 modelleri kaydedildi.")


if __name__ == "__main__":
    train()
    train_v2()
