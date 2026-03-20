from django.urls import path
from . import views

urlpatterns = [
    path('',              views.index,           name='index'),
    path('upload/',       views.upload_page,     name='upload'),
    path('record/',       views.record_page,     name='record'),
    path('api/predict/',  views.predict_disease, name='api_predict'),
    path('api/chat/',     views.chat_api,        name='api_chat'),
    path('api/voice/',    views.voice_api,       name='api_voice'),
    path('api/health/',   views.health_check,    name='api_health'),
    path('api/history/',  views.chat_history,    name='api_history'),
]
