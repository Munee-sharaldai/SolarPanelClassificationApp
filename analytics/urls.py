from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('get-headers/', views.get_csv_headers, name='get_csv_headers'),
    path('signup/', views.signup, name='signup'),
    path('panel/<int:panel_id>/delete/', views.delete_panel, name='delete_panel'),
    path('panel/<int:panel_id>/analysis/', views.panel_analysis, name='panel_analysis'),
    path('panel/<int:panel_id>/update/', views.update_panel_data, name='update_panel_data'),
    path('panel/<int:panel_id>/process_update/', views.process_panel_update, name='process_panel_update'),
]
