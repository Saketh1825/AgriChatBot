from django.contrib import admin
from .models import DiseasePrediction, ChatMessage

@admin.register(DiseasePrediction)
class DiseasePredictionAdmin(admin.ModelAdmin):
    list_display = ('crop', 'disease', 'confidence', 'created_at')
    list_filter  = ('crop',)
    ordering     = ('-created_at',)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('role', 'intent', 'confidence', 'created_at')
    list_filter  = ('role',)
    ordering     = ('-created_at',)
