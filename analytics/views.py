import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from analytics.apps import AnalyticsConfig
from analytics.services import SolarAI_Service
from .models import SolarPanel, Prediction, PanelMetric

from pathlib import Path
import joblib
import tensorflow as tf

# Mapping indices to physical labels as per TS
CLASS_MAPPING = {
    0: "Ночь или Пасмурная погода - Слабая генерация",
    1: "Переменная облачность",
    2: "Критическая аппаратная аномалия - Сбой",
    3: "Переходный режим",
    4: "Ясно - Пиковая мощность"
}

# Color palette
COLOR_MAP = {
    "Ясно / Номинальная генерация": "blue",
    "Переменная облачность": "green",
    "Пасмурно / Слабая генерация": "yellow",
    "Высокая динамика / Переходный режим": "orange",
    "Критическая аппаратная аномалия — Сбой": "red"
}

@login_required
def dashboard(request):
    panels = SolarPanel.objects.filter(user=request.user)
    return render(request, 'analytics/dashboard.html', {'panels': panels})

@login_required
def upload_csv(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        panel_name = request.POST.get('panel_name', 'Default Panel')
        selected_model = request.POST.get('model_choice', 'mlp_model.keras')

        if not uploaded_files:
            return render(request, 'analytics/index.html', {'error': 'Файлы не загружены'})

        processed_count = 0
        errors = []
        panel = None

        for uploaded_file in uploaded_files:
            if not uploaded_file.name.endswith('.csv'):
                errors.append(f"Файл {uploaded_file.name} не является .csv")
                continue

            try:
                df = pd.read_csv(uploaded_file)
                cols = df.columns.tolist()

                # Use explicit mapping from request if available, otherwise fallback to keywords
                poa_col = request.POST.get('map_poa') or next((c for c in cols if 'poa' in c.lower() or 'irradiance' in c.lower()), None)
                temp_col = request.POST.get('map_temp') or next((c for c in cols if 'module' in c.lower() or 'temp' in c.lower()), None)
                power_col = request.POST.get('map_power') or next((c for c in cols if 'ac' in c.lower() or 'power' in c.lower()), None)
                time_col = request.POST.get('map_timestamp') or next((c for c in cols if any(k in c.lower() for k in ['time', 'date', 'дата', 'время', 'timestamp'])), None)

                if not all([poa_col, temp_col, power_col]):
                    errors.append(f"В файле {uploaded_file.name} отсутствуют необходимые колонки")
                    continue

                df_mapped = df.copy()
                df_mapped['poa_irradiance'] = df[poa_col]
                df_mapped['module_temp'] = df[temp_col]
                df_mapped['ac_power'] = df[power_col]

                if time_col:
                    df_mapped['timestamp'] = pd.to_datetime(df[time_col])
                else:
                    df_mapped['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')

                df_mapped = df_mapped.dropna(subset=['poa_irradiance', 'module_temp', 'ac_power'])
                df_mapped = df_mapped[df_mapped['poa_irradiance'] > 10]

                if df_mapped.empty:
                    errors.append(f"Файл {uploaded_file.name} пуст после очистки")
                    continue

                service_df = df_mapped[['timestamp', 'poa_irradiance', 'module_temp', 'ac_power']]
                current_panel, results = SolarAI_Service.process_and_save_data(request.user, panel_name, service_df, model_filename=selected_model)
                panel = current_panel
                processed_count += 1

            except Exception as e:
                errors.append(f"Ошибка при обработке {uploaded_file.name}: {str(e)}")

        if processed_count > 0 and panel:
            return redirect('panel_analysis', panel_id=panel.id)

        if errors:
            return render(request, 'analytics/index.html', {'error': ' | '.join(errors)})

    return render(request, 'analytics/index.html')

@login_required
def get_csv_headers(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            # Read only the header
            df = pd.read_csv(file, nrows=0)
            cols = df.columns.tolist()
            
            # Suggest mappings based on keywords
            suggestions = {
                'timestamp': next((c for c in cols if any(k in c.lower() for k in ['time', 'date', 'дата', 'время', 'timestamp'])), None),
                'poa_irradiance': next((c for c in cols if 'poa' in c.lower() or 'irradiance' in c.lower()), None),
                'module_temp': next((c for c in cols if 'module' in c.lower() or 'temp' in c.lower()), None),
                'ac_power': next((c for c in cols if 'ac' in c.lower() or 'power' in c.lower()), None),
            }
            
            return JsonResponse({'columns': cols, 'suggestions': suggestions})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'No file provided'}, status=400)

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'analytics/signup.html', {'form': form})

@login_required
def update_panel_data(request, panel_id):
    panel = get_object_or_404(SolarPanel, id=panel_id, user=request.user)
    return render(request, 'analytics/update_panel.html', {'panel': panel})

@login_required
def process_panel_update(request, panel_id):
    panel = get_object_or_404(SolarPanel, id=panel_id, user=request.user)
    
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        selected_model = request.POST.get('model_choice', 'mlp_model.keras')

        if not uploaded_files:
            return render(request, 'analytics/update_panel.html', {'panel': panel, 'error': 'Файлы не загружены'})

        processed_count = 0
        errors = []

        for uploaded_file in uploaded_files:
            if not uploaded_file.name.endswith('.csv'):
                errors.append(f"Файл {uploaded_file.name} не является .csv")
                continue

            try:
                df = pd.read_csv(uploaded_file)
                cols = df.columns.tolist()

                # Use explicit mapping from request if available, otherwise fallback to keywords
                poa_col = request.POST.get('map_poa') or next((c for c in cols if 'poa' in c.lower() or 'irradiance' in c.lower()), None)
                temp_col = request.POST.get('map_temp') or next((c for c in cols if 'module' in c.lower() or 'temp' in c.lower()), None)
                power_col = request.POST.get('map_power') or next((c for c in cols if 'ac' in c.lower() or 'power' in c.lower()), None)
                time_col = request.POST.get('map_timestamp') or next((c for c in cols if any(k in c.lower() for k in ['time', 'date', 'дата', 'время', 'timestamp'])), None)

                if not all([poa_col, temp_col, power_col]):
                    errors.append(f"В файле {uploaded_file.name} отсутствуют необходимые колонки")
                    continue

                df_mapped = df.copy()
                df_mapped['poa_irradiance'] = df[poa_col]
                df_mapped['module_temp'] = df[temp_col]
                df_mapped['ac_power'] = df[power_col]

                if time_col:
                    df_mapped['timestamp'] = pd.to_datetime(df[time_col])
                else:
                    df_mapped['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')

                df_mapped = df_mapped.dropna(subset=['poa_irradiance', 'module_temp', 'ac_power'])
                df_mapped = df_mapped[df_mapped['poa_irradiance'] > 10]

                if df_mapped.empty:
                    errors.append(f"Файл {uploaded_file.name} пуст после очистки")
                    continue

                service_df = df_mapped[['timestamp', 'poa_irradiance', 'module_temp', 'ac_power']]
                SolarAI_Service.process_and_save_data(request.user, panel.name, service_df, model_filename=selected_model)
                processed_count += 1

            except Exception as e:
                errors.append(f"Ошибка при обработке {uploaded_file.name}: {str(e)}")

        if processed_count > 0:
            return redirect('panel_analysis', panel_id=panel.id)
        
        if errors:
            return render(request, 'analytics/update_panel.html', {'panel': panel, 'error': ' | '.join(errors)})

    return redirect('panel_analysis', panel_id=panel.id)

@login_required
def delete_panel(request, panel_id):
    panel = get_object_or_404(SolarPanel, id=panel_id, user=request.user)
    if request.method == 'POST':
        panel.delete()
        return redirect('dashboard')
    return render(request, 'analytics/delete_confirm.html', {'panel': panel})

@login_required
def panel_analysis(request, panel_id):
    panel = get_object_or_404(SolarPanel, id=panel_id, user=request.user)

    # Filters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    # Fetch metrics and predictions specifically for this panel to avoid data leakage
    metrics = PanelMetric.objects.filter(panel=panel).order_by('timestamp')

    if start_date:
        metrics = metrics.filter(timestamp__date__gte=start_date)
    if end_date:
        metrics = metrics.filter(timestamp__date__lte=end_date)
    if not metrics.exists():
        return render(request, 'analytics/panel_analysis.html', {
            'panel': panel,
            'error': 'Для выбранного периода нет данных анализа.',
            'start_date': start_date,
            'end_date': end_date
        })

    # Prepare data for plotting
    data = []
    for m in metrics:
        try:
            pred_val = m.prediction.predicted_value
            state = CLASS_MAPPING.get(int(pred_val), "Неизвестный статус")
        except Prediction.DoesNotExist:
            state = "Нет предсказания"
            
        data.append({
            'timestamp': m.timestamp,
            'poa_irradiance': m.irradiance,
            'module_temp': m.temperature,
            'ac_power': m.energy_production,
            'state': state
        })
    
    df = pd.DataFrame(data)
    
    total_points = len(df)
    anomaly_count = (df['state'] == CLASS_MAPPING[2]).sum()
    
    # 3D Scatter Plot
    fig_3d = px.scatter_3d(
        df,
        x='poa_irradiance',
        y='module_temp',
        z='ac_power',
        color='state',
        color_discrete_map=COLOR_MAP,
        labels={
            'poa_irradiance': 'Инсоляция (Вт/м²)',
            'module_temp': 'Температура (°C)',
            'ac_power': 'Мощность (кВт)'
        },
        title=f"Анализ панели: {panel.name}",
        opacity=0.7
    )
    
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
        scene=dict(xaxis_title='Инсоляция', yaxis_title='Температура', zaxis_title='Мощность')
    )
    
    plot_div_3d = pio.to_html(fig_3d, full_html=False, include_plotlyjs='cdn')
    
    # Pie Chart for state distribution
    state_counts = df['state'].value_counts()
    fig_pie = px.pie(
        values=state_counts.values, 
        names=state_counts.index, 
        color=state_counts.index,
        color_discrete_map={state: color for state, color in COLOR_MAP.items() if state in state_counts.index},
        title="Распределение режимов работы"
    )
    
    fig_pie.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    plot_div_pie = pio.to_html(fig_pie, full_html=False, include_plotlyjs=False)
    
    # Generate HTML table for raw data
    table_df = df.copy()
    table_df = table_df.rename(columns={
        'timestamp': 'Время',
        'poa_irradiance': 'Инсоляция (Вт/м²)',
        'module_temp': 'Температура (°C)',
        'ac_power': 'Мощность (кВт)',
        'state': 'Режим работы'
    })
    # Format timestamp for better readability
    table_df['Время'] = table_df['Время'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Convert to HTML table with Bootstrap classes
    plot_div_table = table_df.to_html(classes='table table-sm table-striped table-hover', index=False, border=0)
    
    return render(request, 'analytics/panel_analysis.html', {
        'panel': panel,
        'plot_div_3d': plot_div_3d,
        'plot_div_pie': plot_div_pie,
        'plot_div_table': plot_div_table,
        'total_points': total_points,
        'anomaly_count': anomaly_count,
        'has_anomalies': anomaly_count > 0,
        'start_date': start_date,
        'end_date': end_date,
    })
