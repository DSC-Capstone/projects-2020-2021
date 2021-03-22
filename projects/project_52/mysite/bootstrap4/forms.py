# Brian Cheng
# Eric Liu
# Brent Min

# this file contains the django form used on the website to get user information
# the form is not displayed using the django default, rather index.html nicely lays out the form
# elements using bootstrap 4

from django import forms

class NoColon(forms.Form):
    """
    A simple class that extends djangos defualt form class. Removes the colon that is added to the
    end of form input labels by default.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('label_suffix', '')
        super(NoColon, self).__init__(*args, **kwargs)

class RecInputForm(NoColon):
    """
    All form elements. For some elements, form validation is done using the Field validation 
    arguments such as min_value. For other elements, form validation is done using the 
    secondary_validation function in views.py.
    """
    url = forms.URLField(label="Mountain Project URL:", max_length=100, required=False)
    latitude = forms.DecimalField(label="Latitude:", initial=33.8734, min_value=-90, max_value=90)
    longitude = forms.DecimalField(label="Longitude:", initial=-115.9010, min_value=-180, 
        max_value=180)
    max_distance = forms.IntegerField(label="Max Distance (mi):", initial=50, min_value=1)
    rec = forms.MultipleChoiceField(label="Recommenders:", choices=(
        ("top_pop", "Top Popular"),
        ('cosine_rec', 'Personalized'),))
        # ("debug", "Debug (show I/O)"),))
    num_recs = forms.IntegerField(label="Number of Recommendations:", initial=10, min_value=1)
    boulder_lower = forms.IntegerField(label="V", min_value=0, max_value=16, 
        initial=0)
    boulder_upper = forms.IntegerField(label="V", min_value=0, max_value=16,
        initial=3)
    get_boulder = forms.BooleanField(label="Boulder:", initial=True, required=False)
    route_lower = forms.CharField(label="5.", max_length=3, initial="8")
    route_upper = forms.CharField(label="5.", max_length=3, initial="10d")
    get_route = forms.BooleanField(label="Route:", initial=True, required=False)