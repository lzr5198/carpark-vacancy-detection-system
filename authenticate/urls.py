from django.urls import path
from django.http import HttpResponse
from . import views
urlpatterns = [
    path('',views.loginPage,name="login"),
    path('logout/',views.logoutUser,name="logout"),
    path('home/',views.home,name="home")
]