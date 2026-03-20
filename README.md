# 🌱 AgriBot — AI-Based Crop Disease Detection & Agricultural Chatbot

## 📌 Overview

AgriBot is an AI-powered web application designed to assist farmers and agriculture enthusiasts in detecting crop diseases from leaf images and receiving actionable treatment suggestions. The system integrates a pre-trained deep learning model with a Django-based web interface and includes a chatbot for answering agricultural queries.

This project demonstrates the integration of Machine Learning with full-stack web development for real-world agricultural applications.

---

## 🚀 Key Features

- 📸 **Leaf Image Disease Detection** using Deep Learning
- 🌿 **Crop & Disease Classification** with confidence score
- 💊 **Remedy Suggestions** for identified diseases
- 💬 **Interactive Chatbot** for agriculture-related queries
- 📊 **Database Storage** for predictions and chat history
- 🎨 **Modern User Interface** with real-time interaction

---

## 🛠️ Tech Stack

- **Backend:** Django (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** CNN Model (model.json + model_weights.h5)
- **Database:** SQLite
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
AgriChatbot/
├── model/                  # Pre-trained ML model files
├── ChatBotApp/             # Main Django app
│   ├── models.py
│   ├── views.py
│   ├── templates/
│   ├── static/
│   └── data/
├── Chatbot/                # Project settings
├── db.sqlite3
├── requirements.txt
├── manage.py
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Saketh1825/AgriChatBot.git
cd AgriChatBot
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Model Files

Place your trained model files inside:

```
model/
├── model.json
├── model_weights.h5
```

### 4️⃣ Run Migrations

```bash
python manage.py migrate
```

### 5️⃣ Start Server

```bash
python manage.py runserver
```

### 6️⃣ Open Application

```
http://127.0.0.1:8000
```

---

## 🧠 How It Works

1. User uploads a leaf image
2. The CNN model processes the image
3. The system predicts crop type and disease
4. Confidence score is generated
5. Relevant remedies are displayed
6. Chatbot handles additional user queries

---

## 📸 Screenshots

### 🏠 Home Interface

![Home](screenshots/home.png)

### 💬 Chatbot Interface

![Chat](screenshots/chat.png)

### 📸 Disease Prediction Result

![Prediction](screenshots/prediction.png)

---

## 🎯 Use Cases

- Crop disease detection for farmers
- Agricultural education and research
- Smart farming solutions
- AI-based advisory systems

---

## 📌 Future Enhancements

- 🌐 Cloud deployment (Render / AWS)
- 📱 Mobile application integration
- 🌍 Multi-language chatbot support
- ☁️ Weather API integration
- 📊 Advanced analytics dashboard

---

## 👨‍💻 Author

**Saketh Goudi**
B.Tech CSE (Data Science)
CMR College of Engineering & Technology

---

## ⭐ Acknowledgement

This project is developed for academic and learning purposes to explore AI, Machine Learning, and Web Development integration.

---

## 📬 Contact

Feel free to connect for collaboration or suggestions.
