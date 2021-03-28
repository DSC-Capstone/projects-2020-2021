# Brian Cheng
# Eric Liu
# Brent Min

# views.py contains the logic needed to do form validation and render the various webpages of
# the heroku app

from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.validators import URLValidator

from .forms import RecInputForm

import os

import sys
sys.path.append('../../..')

def about(request):
    return render(request, "about.html")

def developers(request):
    return render(request, "developers.html")

def algorithims(request):
    return render(request, "algorithims.html")

def template(form=None, notes="", latitude=33.8734, longitude=-115.9010, results=[]):
    """
    A nice way to update all template inputs to the render functions all at once.

    :param:     form        The current contents of the main form. This input allows the site to
                            remember what the user has previously entered
    :param:     notes       Various notes to display to the user. Commonly used for debug messages
                            and reporting errors from form validation
    :param:     latitude    The latitude to display on the map. Default is at JTree
    :param:     longitude   The longitude to display on the map. Default is at JTree
    :param:     results     A list containing the recommendations. This is formatted by django
                            templates

    :return:    dict        A dictionary as shown below 
    """
    template_default = {
        "form": form,
        "notes": notes,
        "latitude": latitude,
        "longitude": longitude,
        "results": results,
        "google_maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY")
    }

    return template_default

def bootstrap4_index(request):
    # enter if the button is pressed on the website
    if(request.method == "POST"):

        form = RecInputForm(request.POST)

        if(form.is_valid()):

            # run the secondary validation code
            inputs = secondary_validation(form)

            # if there are errors, then the bool flag would be true
            if(inputs[1]):
                return render(request, 'index.html', template(form, inputs[1], 
                    inputs[0]["location"][0], inputs[0]["location"][1]))

            # run the main code
            from run import main
            results = main(inputs[0])

            # # transform the return dictionary into the proper format for django templates
            # trans_results = format_django(results)

            # return the value of the main code
            return render(request, 'index.html', template(form, results["notes"], 
                inputs[0]["location"][0], inputs[0]["location"][1], results["recommendations"]))

        return render(request, 'index.html', template(form))

    # note on opening the website, set the initial recommender to be top_pop
    form = RecInputForm(initial={"rec": "top_pop"})
    return render(request, 'index.html', template(form))

def secondary_validation(form):  
    """
    This function runs some secondary validation code that I could not integrate into django
    without it messing up the website style

    :param:     form            The form containing cleaned data

    :return:    (dict, str)     The dict contains the input to the main function, and the string 
                                contains the error message (can be "")
    """
    # store error string here if necessary
    errors = []

    # get the url
    url = form.cleaned_data["url"]

    # if top popular recommender is chosen, don't enter and don't validate url
    if not (form.cleaned_data["rec"][0]=="top_pop" or form.cleaned_data["rec"][0]=="debug"):
        if url == '':
            errors.append(f"Must input a Mountain Project user URL")
        else:
            # validate the url structure
            validator = URLValidator()
            try:
                validator(url)
            except ValidationError:
                errors.append(f"Mountain Project URL ({url}) is not a valid user page.")

            # validate that the url contains both "mountainproject.com" and "user"
            if((len(errors) == 0) and (("mountainproject.com" not in url) or ("user" not in url))):
                errors.append(f"Mountain Project URL ({url}) is not a valid user page.")

    # get the boulder grades
    if(form.cleaned_data["get_boulder"]):
        bl = int(form.cleaned_data["boulder_lower"])
        bu = int(form.cleaned_data["boulder_upper"])

        # validate the boulder grades if the box is checked
        if(bl > bu):
            error_str = f"Lowest Boulder Grade (V{bl}) should be less than or equal to Highest " \
                f"Boulder Grade (V{bu})."
            errors.append(error_str)
    # if the user did not want boulders
    else:
        bl = -1
        bu = -1

    # get the route grades
    if(form.cleaned_data["get_route"]):
        rl = route_to_int(form.cleaned_data["route_lower"])
        ru = route_to_int(form.cleaned_data["route_upper"])

        # validate the route grades
        if(rl is None):
            error_str = f"Lowest Route Grade (5.{form.cleaned_data['route_lower']}) is an " \
                "invalid difficulty."
            errors.append(error_str)
        if(ru is None):
            error_str = f"Highest Route Grade (5.{form.cleaned_data['route_upper']}) is an " \
                "invalid difficulty.\n"
            errors.append(error_str)
        if((rl is not None) and (ru is not None)):
            if(rl > ru):
                error_str = f"Lowest Route Grade (5.{form.cleaned_data['route_lower']}) should " \
                    "be less than or equal to Highest Route Grade " \
                    f"(5.{form.cleaned_data['route_upper']}).\n"
                errors.append(error_str)
    # if the user did not want routes
    else:  
        rl = -1
        ru = -1

    # make sure that the user selected at least one of boulder/route
    if(bl == -1 and rl == -1):
        errors.append("One of Boulder or Route must be checked.\n")

    # create the config dictionary to pass into main
    inputs = {
        "user_url": form.cleaned_data["url"],
        "location": [form.cleaned_data["latitude"], form.cleaned_data["longitude"]],
        "max_distance": form.cleaned_data["max_distance"],
        "recommender": form.cleaned_data["rec"][0], # note for some reason ["rec"] is a list
        "num_recs": form.cleaned_data["num_recs"],
        "difficulty_range": {
            "boulder": [bl, bu],
            "route": [rl, ru]
        }
    }
    return (inputs, errors)

def route_to_int(route_str):
    """
    This function takes a route string and turns it into an integer

    :param:     route_str   The stuff after the "5.". Can be anything from "1" to "15d"

    :return:    int         An integer representation of the grade
    """
    mapping = ['3rd', '4th', 'Easy 5th', '0', "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10a", "10b", "10c", "10d", "11a", "11b", "11c", "11d", "12a", "12b", "12c", "12d", "13a", 
        "13b", "13c", "13d", "14a", "14b", "14c", "14d", "15a", "15b", "15c", "15d"]
    if route_str[-1] == '+' or route_str[-1] == '-':
        route_str = route_str[:-1]
    if route_str == '10':
        route_str = '10a'
    if route_str == '11':
        route_str = '11a'
    if route_str == '12':
        route_str = '12a'
    if route_str == '13':
        route_str = '13a'
    if route_str == '14':
        route_str = '14a'
    if route_str == '15':
        route_str = '15a'

    try:
        return mapping.index(route_str.lower())
    except ValueError:
        return None
