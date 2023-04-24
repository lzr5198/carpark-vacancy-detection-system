from django.urls import path
from django.http import HttpResponse
from . import views
urlpatterns = [
    path('spot_list/', views.spot_list, name="spot_list"),
    path('draw/', views.draw, name="draw"),
    path('spot/', views.spot, name='spot'),
    path('undistort/', views.undistort, name='undistort')
]