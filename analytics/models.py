from django.db import models
from django.contrib.auth.models import User

class SolarPanel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='panels')
    name = models.CharField(max_length=100, verbose_name="Название панели")
    location = models.CharField(max_length=255, verbose_name="Местоположение")
    capacity = models.FloatField(verbose_name="Мощность (Вт)")
    installed_date = models.DateField(null=True, blank=True, verbose_name="Дата установки")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    class Meta:
        verbose_name = "Солнечная панель"
        verbose_name_plural = "Солнечные панели"

class PanelMetric(models.Model):
    panel = models.ForeignKey(SolarPanel, on_delete=models.CASCADE, related_name='metrics')
    timestamp = models.DateTimeField(verbose_name="Время замера")
    energy_production = models.FloatField(verbose_name="Выработка энергии (кВтч)")
    temperature = models.FloatField(verbose_name="Температура (°C)")
    irradiance = models.FloatField(verbose_name="Интенсивность излучения (Вт/м²)")
    voltage = models.FloatField(null=True, blank=True, verbose_name="Напряжение (В)")
    current = models.FloatField(null=True, blank=True, verbose_name="Ток (А)")

    def __str__(self):
        return f"Metric for {self.panel.name} at {self.timestamp}"

    class Meta:
        verbose_name = "Показатель панели"
        verbose_name_plural = "Показатели панелей"

class Prediction(models.Model):
    metric = models.OneToOneField(PanelMetric, on_delete=models.CASCADE, related_name='prediction')
    predicted_value = models.FloatField(verbose_name="Предсказанное значение")
    model_name = models.CharField(max_length=100, verbose_name="Использованная модель")
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.metric.panel.name} at {self.metric.timestamp}"

    class Meta:
        verbose_name = "Предсказание"
        verbose_name_plural = "Предсказания"
