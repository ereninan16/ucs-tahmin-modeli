# UCS Tahmin Modeli — Web Arayüzü

XGBoost + Linear Regression ensemble modeli ile kaya dayanımı (UCS) tahmini yapan FastAPI + React web uygulaması.

**Model:** Eren İNAN · Bitirme Çalışması II · Mayıs 2026

---

## Proje Yapısı

```
ucs-ui/
├── backend/
│   ├── Dockerfile
│   ├── main.py           # FastAPI uygulama
│   ├── train.py          # Model eğitim betiği
│   ├── requirements.txt
│   └── model/            # Docker build'de otomatik oluşur
├── frontend/
│   ├── Dockerfile
│   └── index.html        # React + Tailwind tek sayfa
├── docker-compose.yml
├── .dockerignore
└── README.md
```

---

## Hızlı Başlangıç (Docker Compose)

### Gereksinimler
- Docker Desktop kurulu ve çalışıyor olmalı

### 1. Ayağa Kaldır

```bash
cd ucs-ui
docker compose up --build
```

> İlk build ~3-5 dakika sürer (Python bağımlılıkları + model eğitimi).

### 2. Tarayıcıda Aç

| Servis    | Adres                    |
|-----------|--------------------------|
| Frontend  | http://localhost:3000    |
| Backend   | http://localhost:8000    |
| API Docs  | http://localhost:8000/docs |

### 3. Durdur

```bash
docker compose down
```

---

## Sadece Backend Test (Docker olmadan)

```bash
cd backend

# Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Bağımlılıkları kur
pip install -r requirements.txt

# Modeli eğit
python train.py

# API'yi başlat
uvicorn main:app --reload --port 8000
```

API dökümantasyonu: http://localhost:8000/docs

---

## API Endpoints

### `GET /health`
Servis durum kontrolü.

```bash
curl http://localhost:8000/health
```

**Yanıt:**
```json
{"status": "ok", "models_loaded": true}
```

---

### `POST /predict`
Tek numune için UCS tahmini.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Vp": 5.5, "n": 1.5, "rho": 2.72, "Rn": 57, "Is50": 9.0}'
```

**Giriş parametreleri:**

| Parametre | Açıklama              | Birim  | Aralık       |
|-----------|-----------------------|--------|--------------|
| `Vp`      | P-dalga hızı          | km/s   | 1.0 – 8.0    |
| `n`       | Gözeneklilik          | %      | 0.1 – 35.0   |
| `rho`     | Yoğunluk              | g/cm³  | 1.5 – 3.5    |
| `Rn`      | Schmidt çekici        | —      | 10 – 80      |
| `Is50`    | Nokta yükleme indeksi | MPa    | 0.1 – 20.0   |

**Yanıt:**
```json
{
  "ucs_xgboost": 187.3,
  "ucs_linear": 179.6,
  "ucs_ortalama": 183.5,
  "kaya_tipi_tahmin": "Granit",
  "guven": "Yuksek"
}
```

---

### `POST /predict-shap`
Aynı girdi → SHAP parametre etki grafiği (base64 PNG).

```bash
curl -X POST http://localhost:8000/predict-shap \
  -H "Content-Type: application/json" \
  -d '{"Vp": 5.5, "n": 1.5, "rho": 2.72, "Rn": 57, "Is50": 9.0}'
```

**Yanıt:**
```json
{"image": "data:image/png;base64,..."}
```

---

## Tipik Parametre Aralıkları

| Kaya Tipi   | Vp (km/s) | n (%)  | rho (g/cm³) | Rn    | Is50 (MPa) | UCS (MPa)  |
|-------------|-----------|--------|-------------|-------|------------|------------|
| Granit      | 4.5–6.5   | 0.5–3  | 2.60–2.80   | 50–65 | 5–12       | 100–250    |
| Kireçtaşı   | 3.0–5.5   | 2–12   | 2.40–2.70   | 35–55 | 2–7        | 30–150     |
| Kumtaşı     | 2.5–4.5   | 8–25   | 2.20–2.55   | 25–45 | 1–5        | 20–100     |

---

## Sorun Giderme

**Frontend "API Bağlantı Yok" gösteriyor:**
```bash
# Backend loglarını kontrol et
docker compose logs backend

# Sadece backend'i yeniden başlat
docker compose restart backend
```

**Model yüklenemedi hatası:**
```bash
# Backend içine girip train.py çalıştır
docker compose exec backend python train.py
```

**Port çakışması (8000 veya 3000 kullanılıyor):**
```yaml
# docker-compose.yml içinde portları değiştir
ports:
  - "8001:8000"   # backend
  - "3001:80"     # frontend
```

**Build'i temizden başlat:**
```bash
docker compose down --volumes
docker compose up --build --force-recreate
```
