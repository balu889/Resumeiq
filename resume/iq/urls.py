from django.urls import path
from . import views
from django.contrib import admin

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path("admin/", admin.site.urls, name='admin'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('analyze/resume/', views.analyze_resume, name='analyze_resume'),
    path('results/', views.results, name='results'),
]
