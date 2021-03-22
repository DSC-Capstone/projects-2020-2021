from django.shortcuts import render
from django.http import HttpResponse
import os
import json
import pandas as pd
import subprocess
from json import dumps 
from django.views.decorators.csrf import csrf_exempt
import time
import datetime
from django.utils import timezone
from django.contrib.sessions.models import Session

def index(request):
    data = os.listdir('autolibrary/documents')
    data = dumps(data) 
    
    os.system('mkdir -p ../data/raw')
    os.system('mkdir -p ../data/out')
    os.system('mkdir -p static/autolibrary/documents')
    os.system('mkdir -p static/autolibrary/web_scrap')

    shared_obj = request.session.get('myobj',{}) 
    shared_obj['selected_doc'] = ''
    shared_obj['selected_pdf'] = ''
    shared_obj['if_customized'] = "true"
    shared_obj['selected_domain'] = ''
    shared_obj['selected_subdomain'] = ''
    shared_obj['selected_keywords'] = ''
    shared_obj['phrases'] = []
    shared_obj['in_queue'] = "false"
    shared_obj['timestamp'] = ''
    shared_obj['first_run'] = "true"
    request.session['myobj'] = shared_obj

    return render(request, 'autolibrary/index.html', {"data": data})

def result(request):
    data = os.listdir('autolibrary/documents')
    domains = json.load(open('../config/domains_full.json'))

    shared_obj = request.session['myobj']
    selected_doc = shared_obj['selected_doc']
    selected_pdf = shared_obj['selected_pdf']

    content = {
        "data": dumps(data), 
        "selected_doc": dumps([selected_doc]), 
        "selected_pdf": dumps([selected_pdf]), 
        "domains": dumps(domains)
    }
    
    shared_obj['in_queue'] = "false"
    request.session['myobj'] = shared_obj
    return render(request, 'autolibrary/result.html', content)

def customization(request):
    data = os.listdir('autolibrary/documents')
    domains = json.load(open('../config/domains_full.json'))

    shared_obj = request.session['myobj']
    if_customized = shared_obj['if_customized']
    selected_pdf = shared_obj['selected_pdf']
    selected_doc = shared_obj['selected_doc']
    selected_keywords = shared_obj['selected_keywords']
    if shared_obj['first_run'] == "true":
        shared_obj['selected_domain'] = ''
        shared_obj['selected_subdomain'] = ''
        shared_obj['phrases'] = []
    selected_domain = shared_obj['selected_domain']
    selected_subdomain = shared_obj['selected_subdomain']
    phrases = shared_obj['phrases']

    content = {
        "customized": dumps([if_customized]),
        "data": dumps(data), 
        "selected_doc": dumps([selected_doc]), 
        "selected_pdf": dumps([selected_pdf]), 
        "domains": dumps(domains),
        "domain": dumps([selected_domain]),
        "subdomain": dumps([selected_subdomain]),
        "phrases": dumps(phrases),
        "keywords":dumps([selected_keywords]),
    }
    if if_customized == "false":
        if_customized = "true"
        shared_obj['if_customized'] = if_customized

    shared_obj['in_queue'] = "false"
    request.session['myobj'] = shared_obj
    return render(request, 'autolibrary/customization.html', content)

@csrf_exempt
def get_file(request):
    if request.method == 'POST':
        if "file_name" in request.POST:
            shared_obj = request.session['myobj']
            if_customized = "false"
            shared_obj['if_customized'] = if_customized

            # rename document
            file_name = request.POST['file_name']
            pdfname = file_name.replace("'", "")
            pdfname = pdfname.replace(" ", "_")
            os.system('bash autolibrary/rename.sh')
            # save doc name and move to static
            selected_doc = file_name
            selected_pdf = pdfname
            shared_obj['selected_pdf'] = selected_pdf
            shared_obj['selected_doc'] = selected_doc

            command = 'cp autolibrary/documents_copy/' + pdfname + ' static/autolibrary/documents'
            os.system(command)

            shared_obj['in_queue'] = "false"
            shared_obj['first_run'] = "true"
            request.session['myobj'] = shared_obj
            return HttpResponse('get file')
    return HttpResponse('fail to get file')

@csrf_exempt
def get_domain(request): 
    if request.method == 'POST':
        if "domain" in request.POST:
            shared_obj = request.session['myobj'] 
            unique_key = request.session.session_key
            
            # save selected domain to data/out
            selected_domain = request.POST['domain']
            selected_subdomain = request.POST['subdomain']
            selected_pdf = shared_obj['selected_pdf']
            if selected_domain == '':
                selected_domain = 'ALL'
            if selected_subdomain == '':
                selected_subdomain = 'ALL'

            shared_obj['selected_domain'] = selected_domain 
            shared_obj['selected_subdomain'] = selected_subdomain

            # with open('../data/out/selected_domain_' + unique_key + '.txt', 'w') as fp:
            with open('../data/out/selected_domain.txt', 'w') as fp:
                fp.write(selected_subdomain)
            config = {'fos': [selected_domain]}
            # with open('../data/out/fos_' + unique_key + '.json', 'w') as fp:
            with open('../data/out/fos.json', 'w') as fp:
                json.dump(config, fp)
            # rewrite data-params.json
            config = json.load(open('../config/data-params.json'))
            config['pdfname'] = selected_pdf
            with open('autolibrary/data-params.json', 'w') as fp:
                json.dump(config, fp)
            with open('autolibrary/run.sh', 'w') as rsh:
                # move selected document to data/raw
                rsh.write('''cp autolibrary/documents_copy/''')
                rsh.write(selected_pdf)
                rsh.write(''' ../data/raw \n''')
                # move new data-params.json to config
                rsh.write('''cp autolibrary/data-params.json  ../config \n''')
                # run all targets
                rsh.write('''cd .. \n''')
                rsh.write('''python run.py data \n''')
                rsh.write('''python run.py autophrase \n''')
                rsh.write('''python run.py weight ''' + unique_key +  '''\n''')
                rsh.write('''python run.py webscrape ''' + unique_key +  '''\n''')
                rsh.write('''cp data/out/scraped_AutoPhrase.json website/static/autolibrary/web_scrap/scraped_AutoPhrase.json \n''')
            process = subprocess.Popen(['bash', 'autolibrary/run.sh'])
            process.wait()

            # display phrases with a weighted quality score > 0.5
            data = pd.read_csv('../data/out/weighted_AutoPhrase.csv', index_col = "Unnamed: 0")
            phrases = data[data['score'] > 0.5]['phrase'].to_list()
            if len(phrases) < 5:
                phrases = data['phrase'][:5].to_list()
            shared_obj['phrases'] = phrases 

            new_keywords = phrases[0] + ', ' + phrases[1] + ', ' + phrases[2] + ', '
            shared_obj['selected_keywords'] = new_keywords

            shared_obj['in_queue'] = "false"
            shared_obj['first_run'] = "false"
            request.session['myobj'] = shared_obj
            return HttpResponse('get domain')
    return HttpResponse('fail to get domain')

@csrf_exempt
def get_keywords(request):  
    if request.method == 'POST':
        if "keywords" in request.POST:
            shared_obj = request.session['myobj'] 
            # save selected keywords to data/out
            selected_keywords = request.POST['keywords']
            shared_obj['selected_keywords'] = selected_keywords
            config = {'keywords': selected_keywords}
            with open('../data/out/selected_keywords.json', 'w') as fp:
                json.dump(config, fp)
            with open('autolibrary/run.sh', 'w') as rsh:
                # display new webscrape result
                rsh.write('''cd .. \n''')
                rsh.write('''python run.py webscrape \n''')
                rsh.write('''cp data/out/scraped_AutoPhrase.json website/static/autolibrary/web_scrap/scraped_AutoPhrase.json''')
            process = subprocess.Popen(['bash', 'autolibrary/run.sh'])
            process.wait()

            shared_obj['in_queue'] = "false"
            request.session['myobj'] = shared_obj
            return HttpResponse('get keywords')
    return HttpResponse('fail to get keywords')