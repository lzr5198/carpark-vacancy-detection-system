from django.urls import path
from django.http import HttpResponse
from . import views
urlpatterns = [
    path('',views.loginPage,name="login"),
    path('logout/',views.logoutUser,name="logout"),
    path('home/',views.home,name="home"),
    path('draw_rectangles/',views.draw_rectangles,name="draw_rectangles"),
    path('draw/',views.draw,name="draw"),
]