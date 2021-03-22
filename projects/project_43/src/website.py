import os
import platform


def activate_website(key, address):
    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>> Activating website... <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("  => Adding SECRET_KEY back to settings.py...")
    with open('website/website/settings.py', 'r') as f:
        last_line = f.readlines()[-1]
        
    system_name = platform.system()    
    print("  => Starting the Django development server...")
    with open('src/website.sh', 'w') as rsh:
#        rsh.write('''cd website/website \n''')
#        if last_line[:10] != 'SECRET_KEY':
#            rsh.write('''echo "''')
#            rsh.write(key)
#            rsh.write('''" >> settings.py \n''')
#        rsh.write('''cd .. \n''')
        rsh.write('''cd website \n''')
        rsh.write('''python manage.py migrate \n''')
    
        if system_name == 'Darwin':
            rsh.write('''open "http://127.0.0.1:8000/autolibrary" && python manage.py runserver \n''')
        elif system_name == 'Windows':
            rsh.write('''explorer "http://127.0.0.1:8000/autolibrary" && python manage.py startapp autolibrary \n''')
        else:
            print("ubuntu")
            rsh.write('''python -m webbrowser "http://127.0.0.1:8000/autolibrary" && python manage.py runserver \n''')

    os.system('bash src/website.sh')
    return
