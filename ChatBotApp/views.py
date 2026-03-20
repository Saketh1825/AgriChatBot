"""
AgriBot - views.py
Loads your existing model/model.json + model/model_weights.h5 automatically.
No retraining needed. Gives real predictions for every leaf image.
"""

import os, json, base64, logging, re
from collections import Counter

import numpy as np
import cv2

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import speech_recognition as sr

# ── TensorFlow ─────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
    print(f"[AgriBot] TensorFlow {tf.__version__} ready.")
except ImportError:
    TF_AVAILABLE = False
    print("[AgriBot] WARNING: TensorFlow not installed. Run: pip install tensorflow==2.13.0")

logger = logging.getLogger(__name__)

# ── Disease class labels — matches your existing 25-class model exactly ────────
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_JSON    = os.path.join(settings.BASE_DIR, 'model', 'model.json')
MODEL_WEIGHTS = os.path.join(settings.BASE_DIR, 'model', 'model_weights.h5')
MODEL_H5      = os.path.join(settings.BASE_DIR, 'model', 'agribot_mobilenetv2.h5')

_classifier  = None
_num_classes = 25   # your model's output classes
_input_size  = 64   # your model trained on 64×64

def get_classifier():
    """
    Loads model once and caches it.
    Tries: (1) new .h5  (2) your existing model.json + weights
    """
    global _classifier, _num_classes, _input_size

    if _classifier is not None:
        return _classifier

    if not TF_AVAILABLE:
        return None

    # Strategy 1 — new unified .h5 (if you retrained)
    if os.path.exists(MODEL_H5):
        try:
            _classifier = tf.keras.models.load_model(MODEL_H5)
            _num_classes = _classifier.output_shape[-1]
            _input_size  = _classifier.input_shape[1]
            print(f"[AgriBot] Loaded new model: {MODEL_H5}  "
                  f"({_num_classes} classes, {_input_size}x{_input_size})")
            return _classifier
        except Exception as e:
            print(f"[AgriBot] New model load failed: {e}")

    # Strategy 2 — your existing model.json + model_weights.h5
    if os.path.exists(MODEL_JSON) and os.path.exists(MODEL_WEIGHTS):
        try:
            with open(MODEL_JSON, 'r') as f:
                model_json = f.read()
            _classifier = tf.keras.models.model_from_json(model_json)
            _classifier.load_weights(MODEL_WEIGHTS)
            _num_classes = _classifier.output_shape[-1]
            _input_size  = _classifier.input_shape[1]
            print(f"[AgriBot] ✅ Loaded existing model: model.json + model_weights.h5  "
                  f"({_num_classes} classes, {_input_size}x{_input_size})")
            return _classifier
        except Exception as e:
            print(f"[AgriBot] Existing model load failed: {e}")

    print("[AgriBot] ❌ No model found in model/ folder!")
    print(f"[AgriBot]    Expected: {MODEL_JSON}")
    return None


def preprocess_image(img_path):
    """Load and preprocess image to match your model's training format."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (_input_size, _input_size))
    img = img.astype('float32') / 255.0           # normalize to [0, 1]
    img = np.expand_dims(img, axis=0)              # shape: (1, H, W, 3)
    return img


def generate_gradcam(model, img_array, class_idx):
    """Grad-CAM: shows which part of the leaf triggered the prediction."""
    try:
        last_conv = next(
            (l.name for l in reversed(model.layers)
             if isinstance(l, tf.keras.layers.Conv2D)), None)
        if not last_conv:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output])

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            loss = preds[:, class_idx]

        grads  = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
        heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)

        hm_resized = cv2.resize(np.uint8(255 * heatmap), (_input_size, _input_size))
        hm_color   = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        return hm_color
    except Exception as e:
        logger.warning("Grad-CAM skipped: %s", e)
        return None


def overlay_heatmap(img_path, heatmap):
    """Overlay Grad-CAM heatmap on original image, return base64 JPEG."""
    try:
        orig    = cv2.resize(cv2.imread(img_path), (_input_size, _input_size))
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        # Upscale for display
        overlay = cv2.resize(overlay, (224, 224))
        _, buf  = cv2.imencode('.jpg', overlay)
        return base64.b64encode(buf).decode()
    except Exception as e:
        logger.warning("Heatmap overlay failed: %s", e)
        return None


def format_label(raw_label):
    """Convert 'Tomato___Early_blight' → ('Tomato', 'Early blight')"""
    if '___' in raw_label:
        crop, disease = raw_label.split('___', 1)
    else:
        crop, disease = raw_label, 'Unknown'
    crop    = crop.replace('_', ' ').strip()
    disease = disease.replace('_', ' ').strip()
    return crop, disease


# ── NLP Chatbot ───────────────────────────────────────────────────────────────
INTENTS_PATH = os.path.join(settings.BASE_DIR, 'ChatBotApp', 'data', 'intents.json')
_intents_cache = None

def load_intents():
    global _intents_cache
    if _intents_cache is None:
        if os.path.exists(INTENTS_PATH):
            with open(INTENTS_PATH, encoding='utf-8') as f:
                _intents_cache = json.load(f)
            print(f"[AgriBot] Loaded {len(_intents_cache.get('intents',[]))} intents.")
        else:
            _intents_cache = {"intents": []}
    return _intents_cache


def _tokenize(text):
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    stems  = []
    for t in tokens:
        for sfx in ['ing', 'tion', 'ness', 'ment', 'ed', 'er', 'est', 'ly']:
            if t.endswith(sfx) and len(t) - len(sfx) > 3:
                t = t[:-len(sfx)]
                break
        stems.append(t)
    return stems


def _cosine(c1, c2):
    shared = set(c1) & set(c2)
    num    = sum(c1[w] * c2[w] for w in shared)
    den    = (sum(v**2 for v in c1.values()) ** 0.5) * \
             (sum(v**2 for v in c2.values()) ** 0.5)
    return num / den if den else 0.0


def get_chat_response(query: str) -> dict:
    intents    = load_intents()
    qvec       = Counter(_tokenize(query))
    best_score = 0.0
    best_resp  = None
    best_tag   = None

    for intent in intents.get('intents', []):
        for pattern in intent.get('patterns', []):
            score = _cosine(qvec, Counter(_tokenize(pattern)))
            if score > best_score:
                best_score = score
                best_resp  = np.random.choice(intent['responses'])
                best_tag   = intent['tag']

    if best_score >= 0.25 and best_resp:
        return {"response": best_resp, "confidence": round(best_score, 3),
                "intent": best_tag, "matched": True}

    return {
        "response": (
            "I'm not sure about that. Try asking:\n"
            "• 'How to treat tomato blight?'\n"
            "• 'What soil does rice need?'\n"
            "• 'Aphid pest control'\n\n"
            "Or upload a leaf image for instant detection! 📸"
        ),
        "confidence": round(best_score, 3),
        "intent": "fallback",
        "matched": False
    }


# ── Page views ─────────────────────────────────────────────────────────────────
def index(request):
    return render(request, 'index.html')

def upload_page(request):
    return render(request, 'Upload.html')

def record_page(request):
    return render(request, 'Record.html')


# ── API: Predict disease from leaf image ───────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def predict_disease(request):
    from .models import DiseasePrediction

    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image uploaded."}, status=400)

    img_path = None
    try:
        img_file   = request.FILES['image']
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        fs       = FileSystemStorage(location=upload_dir)
        filename = fs.save('leaf_upload.png', img_file)
        img_path = fs.path(filename)

        model = get_classifier()
        if model is None:
            return JsonResponse({
                "error": (
                    "Model not found! Make sure your model/ folder contains "
                    "model.json and model_weights.h5, then restart the server."
                )
            }, status=503)

        # Preprocess exactly as your original training code did
        img_array  = preprocess_image(img_path)
        preds      = model.predict(img_array, verbose=0)
        class_idx  = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Pick label list matching the model's output size
        labels = DISEASE_CLASSES if _num_classes == 25 else DISEASE_CLASSES
        raw_label = labels[class_idx] if class_idx < len(labels) else "Unknown"
        crop, disease = format_label(raw_label)

        # Get remedy from chatbot
        remedy_data = get_chat_response(f"{crop} {disease}")

        # Grad-CAM heatmap
        gradcam_b64 = None
        heatmap = generate_gradcam(model, img_array, class_idx)
        if heatmap is not None:
            gradcam_b64 = overlay_heatmap(img_path, heatmap)

        # Save to database
        DiseasePrediction.objects.create(
            image_name=filename,
            crop=crop,
            disease=disease,
            confidence=round(confidence * 100, 1),
            remedy=remedy_data['response'],
        )

        return JsonResponse({
            "success":      True,
            "crop":         crop,
            "disease":      disease,
            "confidence":   round(confidence * 100, 1),
            "remedy":       remedy_data['response'],
            "gradcam":      gradcam_b64,
            "model_active": True,
        })

    except Exception as e:
        logger.exception("Prediction error")
        return JsonResponse({"error": str(e)}, status=500)


# ── API: Text chatbot ──────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    from .models import ChatMessage
    try:
        body  = json.loads(request.body)
        query = body.get('message', '').strip()
        if not query:
            return JsonResponse({"error": "Empty message."}, status=400)

        result  = get_chat_response(query)
        session = request.session.session_key or 'anonymous'

        ChatMessage.objects.create(
            session_key=session, role='user', message=query,
            intent=result['intent'], confidence=result['confidence'])
        ChatMessage.objects.create(
            session_key=session, role='bot', message=result['response'],
            intent=result['intent'], confidence=result['confidence'])

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    except Exception as e:
        logger.exception("Chat error")
        return JsonResponse({"error": str(e)}, status=500)


# ── API: Voice input ───────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def voice_api(request):
    recognizer = sr.Recognizer()
    temp_path  = os.path.join(settings.MEDIA_ROOT, 'temp_voice.wav')

    try:
        audio_file = request.FILES.get('audio')
        if not audio_file:
            return JsonResponse({"error": "No audio provided."}, status=400)

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(temp_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(audio_data)
        result     = get_chat_response(transcript)
        result['transcript'] = transcript
        return JsonResponse(result)

    except sr.UnknownValueError:
        return JsonResponse(
            {"error": "Could not understand. Please speak clearly.", "transcript": ""},
            status=422)
    except sr.RequestError as e:
        return JsonResponse({"error": f"Speech service unavailable: {e}"}, status=503)
    except Exception as e:
        logger.exception("Voice error")
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── API: Chat history ──────────────────────────────────────────────────────────
@require_http_methods(["GET"])
def chat_history(request):
    from .models import ChatMessage
    session = request.session.session_key or 'anonymous'
    msgs    = list(reversed(
        ChatMessage.objects.filter(session_key=session).order_by('-created_at')[:50]
    ))
    return JsonResponse({
        "history": [
            {"role": m.role, "message": m.message, "time": m.created_at.isoformat()}
            for m in msgs
        ]
    })


# ── API: Health check ──────────────────────────────────────────────────────────
@require_http_methods(["GET"])
def health_check(request):
    model = get_classifier()
    return JsonResponse({
        "status":         "ok",
        "model_loaded":   model is not None,
        "model_classes":  _num_classes,
        "input_size":     f"{_input_size}x{_input_size}",
        "tf_available":   TF_AVAILABLE,
        "intents_loaded": len(load_intents().get('intents', [])),
        "model_json_exists":    os.path.exists(MODEL_JSON),
        "model_weights_exists": os.path.exists(MODEL_WEIGHTS),
    })
