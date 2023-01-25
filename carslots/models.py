from django.db import models

# Create your models here.
class Carslot(models.Model):
    slotId      = models.TextField(default="A1")
    occupied    = models.BooleanField(default=False)
    x1          = models.DecimalField(decimal_places=2, max_digits=7, default=0.0)
    y1          = models.DecimalField(decimal_places=2, max_digits=7, default=0.0)
    x2          = models.DecimalField(decimal_places=2, max_digits=7, default=0.0)
    y2          = models.DecimalField(decimal_places=2, max_digits=7, default=0.0)