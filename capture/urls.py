from django.urls import path
from django.http import HttpResponse
from . import views
urlpatterns = [
    path('record/', views.record, name="record"),
    path('screenshot/', views.screenshot, name="screenshot"),
]