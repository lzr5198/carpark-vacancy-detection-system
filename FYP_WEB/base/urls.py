from django.urls import path
from django.http import HttpResponse
from . import views
urlpatterns = [
    path('login/',views.loginPage,name="login"),
    path('logout/',views.logoutUser,name="logout"),
    path('',views.home,name="home"),
    path('draw_rectangles',views.draw_rectangles,name="draw_rectangles"),
    path('draw/<str:base>',views.draw,name="draw"),
]