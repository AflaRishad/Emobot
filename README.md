# EmoBot

EmoBot is an AI powered Flask web application that delivers personalized reading recommendations based on a user’s emotional state and preferences. The app captures a webcam image, detects the user’s mood using a deep learning model, then starts a short conversational flow to recommend books, manga, or comics that match the user’s current feeling.

The project combines computer vision, conversational AI, and recommendation systems into a single interactive experience.

---

# Features

- Real time emotion detection using a trained Keras model
- Webcam image capture with OpenCV
- AI powered chatbot conversation flow using Ollama
- Personalized recommendations for:
  - Books
  - Manga
  - Comics
- Structured preference extraction using `PREFS_JSON`
- Interactive chat style interface
- Multiple recommendation APIs with fallback systems
- Mood aware recommendation engine
- Local LLM integration through Ollama

---

# Visuals

<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/09e236ef-a498-4b69-b3e6-a338d943d55c" />
<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/386e36e1-c862-46a9-915a-563029497aa3" />
<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/1aeb9a07-4bd4-48a8-b257-e5d087a755ff" />

---

# Tech Stack

## Backend
- Python
- Flask

## Machine Learning
- TensorFlow
- Keras
- OpenCV

## Frontend
- HTML
- CSS
- JavaScript

## AI / LLM
- Ollama
- Phi3 model

## APIs Used
- Google Books API
- OpenLibrary API
- MangaDex API
- Jikan API

---

# Project Structure

```text
EmoBot/
│
├── app.py
├── books.py
├── model.keras
├── config.py
├── requirements.txt
│
├── chatbot/
│   └── chatbot_logic.py
│
├── scraper/
│
├── templates/
│   ├── index.html
│   └── chatbot.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
└── README.md
```

---

# Installation Instructions

## Prerequisites

Before running the project, make sure the following are installed:

- Python 3.10+
- pip
- Ollama installed locally
- Webcam access enabled
- `model.keras` available in the project root

---

## Clone the Repository

```bash
git clone <your-repository-url>
cd EmoBot
```

---

## Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Pull the required model:

```powershell
ollama pull phi3
```

Start the Ollama server:

```powershell
ollama serve
```

Confirm the API endpoint is active:

```text
http://localhost:11434/api/chat
```

If you use a different Ollama model, update the `OLLAMA_MODEL` variable inside `app.py`.

---

# Running the Application

Start the Flask server:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

# Usage Workflow

## Step 1 — Capture Emotion

- Open the landing page
- Allow webcam access
- Capture a photo using the webcam

The image is sent to the Flask backend for emotion analysis.

---

## Step 2 — Emotion Detection

- OpenCV preprocesses the captured image
- The Keras model predicts the emotion label
- The detected mood is stored in the Flask session

Example emotions may include:
- Happy
- Sad
- Neutral
- Angry
- Fear
- Surprise

---

## Step 3 — Chatbot Interaction

After emotion detection:

- The chatbot asks a few short questions
- The user selects:
  - media type
  - genre
  - emotional tone
  - reading preference

The conversation is powered by Ollama using the Phi3 model.

---

## Step 4 — Preference Extraction

The chatbot extracts structured preferences using `PREFS_JSON`.

Example:

```json
{
  "media_type": "manga",
  "genre": "fantasy",
  "tone": "uplifting"
}
```

---

## Step 5 — Recommendation Generation

The application searches multiple APIs and curated fallback lists to generate personalized recommendations.

Recommendations are based on:
- detected emotion
- preferred media type
- selected genre
- emotional tone

---

# Recommendation Sources

| Media Type | Sources |
|---|---|
| Books | Google Books API, OpenLibrary |
| Manga | MangaDex API, Jikan API |
| Comics | Google Books API, curated fallback lists |

Fallback recommendations are used if API results are limited or unavailable.

---

# Core Files Explained

| File | Purpose |
|---|---|
| `app.py` | Main Flask application and recommendation engine |
| `books.py` | Shared search and recommendation utilities |
| `chatbot/chatbot_logic.py` | Additional chatbot logic |
| `templates/index.html` | Webcam capture page |
| `templates/chatbot.html` | Chat interface and recommendations |
| `static/js/camera.js` | Webcam capture logic |
| `static/css/style.css` | Application styling |
| `model.keras` | Emotion classification model |
| `requirements.txt` | Python dependencies |
| `config.py` | Configuration placeholders |

---

# How the System Works

## Emotion Detection Pipeline

1. Webcam captures image
2. Image sent to `/predict`
3. OpenCV preprocesses image
4. Keras model predicts emotion
5. Emotion stored in session

---

## Chatbot Pipeline

1. User message sent to `/chat`
2. Conversation history included in prompt
3. Ollama generates response
4. Preferences extracted using `PREFS_JSON`
5. Recommendation engine triggered

---

## Recommendation Engine

The recommendation engine:
- analyzes detected emotion
- processes user preferences
- queries external APIs
- ranks matching results
- displays recommendation cards in the UI

---

# Ollama Integration

EmoBot uses Ollama as a local LLM backend.

## Current Setup

- Endpoint:
  
```text
http://localhost:11434/api/chat
```

- Model:
  
```text
phi3
```

## Responsibilities of the LLM

- Conduct chatbot conversations
- Ask follow up preference questions
- Extract structured preference data
- Improve recommendation relevance

---

# Deployment Notes

This project is currently designed as a traditional Flask application.

Deploying to serverless platforms like Vercel may require additional setup.

## Possible Deployment Adjustments

- Add `vercel.json`
- Create API wrappers
- Move large ML models to cloud storage
- Replace local Ollama with hosted LLM APIs
- Optimize TensorFlow model loading

---

# Git Guidance

## Recommended Files to Include

```text
app.py
books.py
requirements.txt
templates/
static/
chatbot/
model.keras
config.py
README.md
```

---

## Recommended `.gitignore`

Create a `.gitignore` file:

```text
__pycache__/
.venv/
.env
*.pyc
```

---

# Troubleshooting

## Emotion Detection Issues

Problem:
- Emotion prediction fails

Solutions:
- Verify `model.keras` exists
- Check webcam permissions
- Ensure image quality is clear

---

## Chatbot Issues

Problem:
- Chatbot gives unrelated recommendations

Solutions:
- Verify Ollama is running
- Review prompt logic in `app.py`
- Check `PREFS_JSON` extraction

---

## API Errors

Problem:
- Recommendations fail to load

Solutions:
- Check internet connection
- Verify external APIs are reachable
- Ensure API endpoints are active

---

# Future Improvements

- User authentication
- Reading history tracking
- Voice interaction support
- Better emotion classification accuracy
- More recommendation categories
- Dark mode UI
- Cloud deployment support
- Real time recommendation ranking
- User preference memory
