import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from analytics.models import SolarPanel, PanelMetric, Prediction

class SolarAI_Service:
    _scaler = None
    _models_cache = {}

    @classmethod
    def load_model(cls, model_filename):
        """Загружает модель и скалер. Использует кэш для моделей."""
        base_path = Path(__file__).resolve().parent / 'models_data'
        scaler_path = base_path / 'scaler.joblib'
        model_path = base_path / model_filename

        # Загружаем скалер один раз
        if cls._scaler is None:
            try:
                cls._scaler = joblib.load(scaler_path)
            except Exception as e:
                raise RuntimeError(f"Ошибка загрузки скалера: {e}")

        # Загружаем модель из кэша или с диска
        if model_filename not in cls._models_cache:
            try:
                cls._models_cache[model_filename] = tf.keras.models.load_model(model_path)
            except Exception as e:
                raise RuntimeError(f"Ошибка загрузки модели {model_filename}: {e}")

        return cls._scaler, cls._models_cache[model_filename]

    @classmethod
    def process_and_save_data(cls, user, panel_name, df, model_filename='mlp_model.keras'):
        """
        ОПТИМИЗИРОВАННЫЙ метод обработки данных с выбором модели.
        """
        scaler, model = cls.load_model(model_filename)

        # 1. Создаем или получаем панель
        panel, created = SolarPanel.objects.get_or_create(
            user=user,
            name=panel_name,
            defaults={'location': 'Не указано', 'capacity': 0.0}
        )

        # 2. Подготовка данных для предсказания (Векторизация)
        X = pd.DataFrame(df[['poa_irradiance', 'module_temp', 'ac_power']].values,
                         columns=['poa_irradiance', 'module_temp', 'ac_power'])

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled, verbose=0)
        class_indices = np.argmax(predictions, axis=1)

        # 3. Подготовка объектов для массового сохранения (Bulk Create)
        metrics_to_create = []

        for i, (_, row) in enumerate(df.iterrows()):
            ts = row['timestamp']
            if pd.isnull(ts):
                continue

            try:
                if hasattr(ts, 'tz') and ts.tz is None:
                    ts = timezone.make_aware(ts)
            except Exception:
                pass

            metrics_to_create.append(PanelMetric(
                panel=panel,
                timestamp=ts,
                energy_production=row['ac_power'],
                temperature=row['module_temp'],
                irradiance=row['poa_irradiance']
            ))

        created_metrics = PanelMetric.objects.bulk_create(metrics_to_create)

        # 4. Создаем предсказания, связывая их с созданными метриками
        predictions_to_create = []
        for i, metric in enumerate(created_metrics):
            try:
                state_idx = class_indices[i]
                predictions_to_create.append(Prediction(
                    metric=metric,
                    predicted_value=float(state_idx),
                    model_name=model_filename
                ))
            except IndexError:
                continue

        Prediction.objects.bulk_create(predictions_to_create)

        results = []
        for i, metric in enumerate(created_metrics):
            try:
                results.append((metric, class_indices[i]))
            except IndexError:
                continue

        return panel, results
