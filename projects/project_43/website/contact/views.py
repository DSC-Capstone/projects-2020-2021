from django.shortcuts import render
import os
from json import dumps 

def index(request):
    data = os.listdir('autolibrary/documents')
    data = dumps(data) 
    return render(request, 'contact/index.html', {"data": data})