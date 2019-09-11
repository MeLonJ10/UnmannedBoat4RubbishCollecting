from django.urls import path

from . import views
from . import motion
from . import control

app_name='first'
urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    path('getmotordata',views.getmotordata,name='getmotordata'),
    path('getdirection',control.getdirection, name='getdirection'),
    path('getmode',control.getmode, name='getmode'),
    path('getmotion',motion.getmotion, name='getmotion'),
    ]
