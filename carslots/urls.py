from django.urls import path
from . import views

urlpatterns = [
    path('', views.process_data, name='process_data'),
    path('process_data/', views.process_data, name='process_data'),
    path('ajax/', views.ajax_function, name='ajax_function')
]