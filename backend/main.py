"""
UCS Tahmin Modeli - FastAPI Backend
"""
import base64
import io
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

app = FastAPI(title="UCS Tahmin API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model nesneleri (startup'ta bir kez yuklenir) ---
_xgb = None
_lr = None
_scaler = None
_features = None
_explainer = None


@app.on_event("startup")
def load_models():
    global _xgb, _lr, _scaler, _features, _explainer
    try:
        _xgb = joblib.load(os.path.join(MODEL_DIR, "model_xgboost.pkl"))
        _lr = joblib.load(os.path.join(MODEL_DIR, "model_linear_regression.pkl"))
        _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
        _explainer = shap.TreeExplainer(_xgb)
        print("Modeller yuklendi.")
    except FileNotFoundError as e:
        print(f"Model dosyasi bulunamadi: {e}. train.py'yi calistirin.")


# --- Kaya tipi kural tabanli tahmin ---
def kaya_tipi_tahmin(Vp: float, n: float, rho: float) -> str:
    if n < 4 and Vp > 4.5:
        return "Granit"
    elif n < 13 and Vp > 3.0:
        return "Kirectasi"
    return "Kumtasi"


# --- Pydantic modelleri ---
class TahminGirdi(BaseModel):
    Vp: float = Field(..., ge=1.0, le=8.0, description="P-dalga hizi (km/s)")
    n: float = Field(..., ge=0.1, le=35.0, description="Gozeneklilik (%)")
    rho: float = Field(..., ge=1.5, le=3.5, description="Yogunluk (g/cm3)")
    Rn: float = Field(..., ge=10.0, le=80.0, description="Schmidt cekici geri sicrama")
    Is50: float = Field(..., ge=0.1, le=20.0, description="Nokta yukleme indeksi (MPa)")


class TahminSonuc(BaseModel):
    ucs_xgboost: float
    ucs_linear: float
    ucs_ortalama: float
    kaya_tipi_tahmin: str
    guven: str


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _xgb is not None}


@app.post("/predict", response_model=TahminSonuc)
def predict(girdi: TahminGirdi):
    if _xgb is None:
        raise HTTPException(503, "Model yuklenemedi. Lutfen train.py calistirin.")

    df = pd.DataFrame([girdi.model_dump()])[_features]

    ucs_xgb = float(_xgb.predict(df)[0])
    ucs_lr = float(_lr.predict(_scaler.transform(df))[0])
    ucs_ort = round((ucs_xgb + ucs_lr) / 2, 1)
    fark = abs(ucs_xgb - ucs_lr)
    guven = "Yuksek" if fark < 10 else "Orta" if fark < 20 else "Dusuk"
    kaya = kaya_tipi_tahmin(girdi.Vp, girdi.n, girdi.rho)

    return TahminSonuc(
        ucs_xgboost=round(ucs_xgb, 1),
        ucs_linear=round(ucs_lr, 1),
        ucs_ortalama=ucs_ort,
        kaya_tipi_tahmin=kaya,
        guven=guven,
    )


@app.post("/predict-shap")
def predict_shap(girdi: TahminGirdi):
    if _xgb is None:
        raise HTTPException(503, "Model yuklenemedi.")

    df = pd.DataFrame([girdi.model_dump()])[_features]
    shap_vals = _explainer.shap_values(df)
    base_val = _explainer.expected_value

    fig, ax = plt.subplots(figsize=(9, 4))
    feature_labels = ["Vp (km/s)", "n (%)", "rho (g/cm³)", "Rn", "Is50 (MPa)"]
    vals = shap_vals[0]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    bars = ax.barh(feature_labels, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(
            v + (0.3 if v >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=10, fontweight="bold",
        )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP Degeri (UCS uzerindeki etki, MPa)", fontsize=10)
    ax.set_title(
        f"Parametre Etkileri (Baz deger: {base_val:.1f} MPa)", fontsize=11, fontweight="bold"
    )
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return {"image": f"data:image/png;base64,{encoded}"}
