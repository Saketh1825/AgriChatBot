from django.db import models

class DiseasePrediction(models.Model):
    image_name = models.CharField(max_length=255)
    crop       = models.CharField(max_length=100)
    disease    = models.CharField(max_length=200)
    confidence = models.FloatField()
    remedy     = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.crop} — {self.disease} ({self.confidence}%)"


class ChatMessage(models.Model):
    ROLES = [('user', 'User'), ('bot', 'Bot')]
    session_key = models.CharField(max_length=40, db_index=True)
    role        = models.CharField(max_length=4, choices=ROLES)
    message     = models.TextField()
    intent      = models.CharField(max_length=100, blank=True, null=True)
    confidence  = models.FloatField(default=0.0)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role}] {self.message[:60]}"
