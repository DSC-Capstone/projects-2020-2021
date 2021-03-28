from django.conf.urls import url
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    url("about/", views.about, name="about"),
    url("developers/", views.developers, name="developers"),
    url("algorithms/", views.algorithims, name="algorithms"),
    url("", views.bootstrap4_index, name="index"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)