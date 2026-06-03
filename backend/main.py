"""
UCS Tahmin Modeli - FastAPI Backend
"""
import base64
import io
import json
import os
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

app = FastAPI(title="UCS Tahmin API", version="2.0.0")

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

# v2 modelleri
_v2a = None
_v2b = None
_kaya_turleri: list = []


@app.on_event("startup")
def load_models():
    global _xgb, _lr, _scaler, _features, _explainer
    global _v2a, _v2b, _kaya_turleri
    try:
        _xgb = joblib.load(os.path.join(MODEL_DIR, "model_xgboost.pkl"))
        _lr = joblib.load(os.path.join(MODEL_DIR, "model_linear_regression.pkl"))
        _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
        _explainer = shap.TreeExplainer(_xgb)
        print("v1 modelleri yuklendi.")
    except FileNotFoundError as e:
        print(f"v1 model dosyasi bulunamadi: {e}. train.py'yi calistirin.")

    try:
        _v2a = joblib.load(os.path.join(MODEL_DIR, "model_v2a.pkl"))
        _v2b = joblib.load(os.path.join(MODEL_DIR, "model_v2b.pkl"))
        kt_path = os.path.join(MODEL_DIR, "kaya_turleri.json")
        with open(kt_path, encoding="utf-8") as f:
            _kaya_turleri = json.load(f)
        print(f"v2 modelleri yuklendi. Kaya turleri: {_kaya_turleri}")
    except FileNotFoundError as e:
        print(f"v2 model dosyasi bulunamadi: {e}. train.py'yi calistirin.")


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


# ============================================================
# v2 — Gercek Veri Modeli (SVR, 4 parametre, yogunluk yok)
# ============================================================

def _isrm_sinifi(ucs: float) -> str:
    if ucs < 1:
        return "Son Derece Düşük Dayanımlı (R0)"
    if ucs < 5:
        return "Çok Düşük Dayanımlı (R1)"
    if ucs < 25:
        return "Düşük Dayanımlı (R2)"
    if ucs < 50:
        return "Orta Dayanımlı (R3)"
    if ucs < 100:
        return "Yüksek Dayanımlı (R4)"
    if ucs < 250:
        return "Çok Yüksek Dayanımlı (R5)"
    return "Aşırı Yüksek Dayanımlı (R6)"


class PredictV2Request(BaseModel):
    vp_ms: float = Field(..., ge=500, le=10000, description="P-dalga hizi (m/s)")
    n: float = Field(..., ge=0.01, le=35.0, description="Gozeneklilik (%)")
    shr: float = Field(..., ge=10.0, le=80.0, description="Schmidt rebound (Rn)")
    is50: float = Field(..., ge=0.1, le=20.0, description="Nokta yukleme indeksi (MPa)")
    rock_type: Optional[str] = None


class PredictV2Response(BaseModel):
    ucs_mpa: float
    model: str
    model_r2: float
    isrm_sinifi: str


@app.get("/v2/kaya-turleri")
def get_kaya_turleri():
    return {"kaya_turleri": _kaya_turleri}


@app.post("/v2/predict", response_model=PredictV2Response)
def predict_v2(req: PredictV2Request):
    if _v2a is None:
        raise HTTPException(503, "v2 modeli yuklenemedi. Lutfen train.py calistirin.")

    use_v2b = req.rock_type is not None and req.rock_type in _kaya_turleri

    if use_v2b:
        Xb = pd.DataFrame([{
            "Vp_ms": req.vp_ms,
            "n": req.n,
            "SHR": req.shr,
            "Is50": req.is50,
            "RockType": req.rock_type,
        }])
        ucs = float(_v2b.predict(Xb)[0])
        model_name = "SVR (kaya türlü)"
        r2 = 0.86
    else:
        Xa = np.array([[req.vp_ms, req.n, req.shr, req.is50]])
        ucs = float(_v2a.predict(Xa)[0])
        model_name = "SVR (türsüz)"
        r2 = 0.70

    ucs = round(max(1.0, ucs), 1)
    return PredictV2Response(
        ucs_mpa=ucs,
        model=model_name,
        model_r2=r2,
        isrm_sinifi=_isrm_sinifi(ucs),
    )


@app.post("/v2/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if _v2a is None:
        return JSONResponse(status_code=503, content={"error": "v2 modeli yuklenemedi."})

    contents = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".csv"):
            text = None
            for enc in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
                try:
                    text = contents.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                return JSONResponse(status_code=400, content={"error": "Dosya kodlaması okunamadı."})
            ilk_satir = text.split("\n")[0]
            ayirici = ";" if ilk_satir.count(";") > ilk_satir.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=ayirici)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={
                "error": "Sadece CSV veya Excel (.xlsx/.xls) dosyaları desteklenir."
            })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Dosya okunamadı: {e}"})

    df.columns = [str(c).strip().replace("﻿", "") for c in df.columns]

    zorunlu = ["Vp_ms", "n", "SHR", "Is50"]
    eksik = [c for c in zorunlu if c not in df.columns]
    if eksik:
        return JSONResponse(status_code=400, content={
            "error": f"Eksik sütun(lar): {', '.join(eksik)}. Gerekli: Vp_ms, n, SHR, Is50 (+ opsiyonel RockType)"
        })

    if "RockType" not in df.columns:
        df["RockType"] = ""

    sonuclar = []
    for _, row in df.iterrows():
        try:
            rock = str(row["RockType"]).strip() if pd.notna(row["RockType"]) else ""
            if rock and rock.lower() not in ("", "nan", "none"):
                Xb = pd.DataFrame([{
                    "Vp_ms": float(row["Vp_ms"]), "n": float(row["n"]),
                    "SHR": float(row["SHR"]), "Is50": float(row["Is50"]),
                    "RockType": rock,
                }])
                pred = round(max(1.0, float(_v2b.predict(Xb)[0])), 1)
                kullanilan = "v2b"
            else:
                Xa = np.array([[float(row["Vp_ms"]), float(row["n"]),
                                float(row["SHR"]), float(row["Is50"])]])
                pred = round(max(1.0, float(_v2a.predict(Xa)[0])), 1)
                kullanilan = "v2a"
                rock = ""

            sonuclar.append({
                "Vp_ms": row["Vp_ms"], "n": row["n"],
                "SHR": row["SHR"], "Is50": row["Is50"],
                "RockType": rock if rock else "-",
                "UCS_Tahmin": pred,
                "ISRM_Sinifi": _isrm_sinifi(pred),
                "Model": kullanilan,
            })
        except Exception as e:
            sonuclar.append({
                "Vp_ms": row.get("Vp_ms", "?"), "n": row.get("n", "?"),
                "SHR": row.get("SHR", "?"), "Is50": row.get("Is50", "?"),
                "RockType": str(row.get("RockType", "-")),
                "UCS_Tahmin": None, "ISRM_Sinifi": "HATA", "Model": "-",
                "Hata": str(e),
            })

    return {"satir_sayisi": len(sonuclar), "sonuclar": sonuclar}
