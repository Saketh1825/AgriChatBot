# 🌱 AgriBot — AI Crop Disease Detector

## ⚡ Quick Start (3 steps only)

### Step 1 — Copy your model files

Copy these 2 files from your OLD project into the `model/` folder here:

```
model/model.json          ← copy from old AgriChatbot/model/
model/model_weights.h5    ← copy from old AgriChatbot/model/
```

### Step 2 — Run setup (Windows)

Double-click `SETUP_AND_RUN.bat`

OR run in terminal:

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Step 3 — Open browser

Go to: **http://127.0.0.1:8000**

---

## ✅ What's upgraded

| Before                                 | After                                  |
| -------------------------------------- | -------------------------------------- |
| Django 2.1                             | Django 4.2                             |
| Hardcoded `C:/3/AgriChatbot/...` paths | Fully portable — works on any PC       |
| `cv2.imshow()` in web server (crashed) | Proper image API response              |
| Raw HttpResponse text                  | Clean JSON REST APIs                   |
| No database storage                    | Predictions saved to DB (resume value) |
| Basic HTML                             | Modern drag-drop UI with Grad-CAM      |
| TF-IDF chatbot                         | Intent-based NLP with 40+ crop intents |

---

## 📁 Project Structure

```
AgriChatbot/
├── model/
│   ├── model.json           ← COPY FROM YOUR OLD PROJECT
│   └── model_weights.h5     ← COPY FROM YOUR OLD PROJECT
├── ChatBotApp/
│   ├── views.py             (all AI logic)
│   ├── models.py            (database)
│   ├── data/intents.json    (40+ NLP intents)
│   └── templates/           (3 HTML pages)
├── Chatbot/
│   └── settings.py          (configuration)
├── manage.py
├── requirements.txt
└── SETUP_AND_RUN.bat        ← double-click to start
```

---

## 🔌 API Endpoints

| Endpoint        | Method | What it does                             |
| --------------- | ------ | ---------------------------------------- |
| `/api/predict/` | POST   | Upload leaf → disease + remedy + heatmap |
| `/api/chat/`    | POST   | Text question → NLP answer               |
| `/api/voice/`   | POST   | Voice audio → transcript + answer        |
| `/api/health/`  | GET    | Check if model is loaded correctly       |
| `/api/history/` | GET    | Chat history for this session            |

Check model status: http://127.0.0.1:8000/api/health/
