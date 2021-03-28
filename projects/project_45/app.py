from src.models.light_fm import light_fm, evaluate, pred_i
from src.data.model_preprocessing import get_data
import os
from flask import send_from_directory
from src.app.register import register_user, update_preferences
from flask import Flask, render_template, redirect, url_for, session, g, request
from src.app.forms import RegistrationForm, LoginForm, WorkoutInformation
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import json
import pandas as pd
from src.app.recommendations import create_rec_lists, get_rec_sorted

import sys
sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/models')


app = Flask(__name__)

is_prod = os.environ.get('IS_HEROKU', None)

if is_prod:
    app.config['MYSQL_HOST'] = os.environ.get('MYSQL_HOST')
    app.config['MYSQL_USER'] = os.environ.get('MYSQL_USER')
    app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD')
    app.config['MYSQL_DB'] = os.environ.get('MYSQL_DB')
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
else:
    db_config = json.load(open('./config/db_config.json'))
    flask_keys = json.load(open('./config/flask_keys.json'))
    app.config['MYSQL_HOST'] = db_config['mysql_host']
    app.config['MYSQL_USER'] = db_config['mysql_user']
    app.config['MYSQL_PASSWORD'] = db_config['mysql_password']
    app.config['MYSQL_DB'] = db_config['mysql_db']
    app.config['SECRET_KEY'] = flask_keys['secret_key']



db = MySQL(app)
bcrypt = Bcrypt(app)


@app.before_request
def before_request():
    if 'user_id' in session:
        query = "SELECT * FROM users WHERE user_id = " + \
            str(session['user_id'])
        results = pd.read_sql_query(query, db.connection)
        g.user = results.iloc[0]
    else:
        g.user = None


@app.route('/register', methods=['GET', 'POST'])
def registration_page():
    form = RegistrationForm()
    if form.validate_on_submit():
        cur = db.connection.cursor()

        # check if email already exists in database
        cur.execute("SELECT email FROM users WHERE email = %s",
                    (form.email.data,))
        result = cur.fetchone()
        if result is not None:  # display error
            return render_template('registration_page.html', form=form, email_error=True)
        else:  # insert user into database
            # hash the password
            hashed_password = bcrypt.generate_password_hash(
                form.password.data).decode('utf-8')

            # get next user id
            cur.execute("SELECT MAX(user_id) FROM users")
            result = cur.fetchone()[0]
            if result is None:
                user_id = 5000  # fbcommenters end with id 4026, our users will start from 5000
            else:
                user_id = result + 1

            cur.execute(*register_user(form, user_id, hashed_password))
            db.connection.commit()
            return redirect(url_for('login_page'))
        cur.close()
    return render_template('registration_page.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        cur = db.connection.cursor()

        # check if login information is correct
        cur.execute("SELECT * FROM users WHERE email = %s", (form.email.data,))
        result = cur.fetchone()

        if result is None:  # display error if email doesn't exist
            return render_template('login_page.html', form=form, email_error=True)
        else:  # check password
            db_password = result[3]
            pw_match = bcrypt.check_password_hash(
                db_password, form.password.data)

            if not pw_match:  # display error if password doesn't match
                return render_template('login_page.html', form=form, password_error=True)
            else:  # login, set session to logged in user's id
                session['user_id'] = result[0]
                return redirect(url_for('recommendation_page'))
        cur.close()

    return render_template('login_page.html', form=form)


@app.route('/', methods=['GET', 'POST'])
def recommendation_page():
    # if user is not logged in, redirect to login page
    if g.user is None:
        return redirect(url_for('login_page'))

    rec_engine = request.form.get("engine")
    if rec_engine is None:
        return render_template("recommendation_page.html", dropdown_option=None, rec_dct=None)

    # user's previous interactions
    all_user_interactions = pd.read_sql_query(
        "SELECT * FROM workout.user_item_interaction WHERE user_id = " + str(
            session['user_id']), db.connection
    )

    # user's disliked items
    user_disliked_items = pd.read_sql_query(
        "SELECT * FROM workout.user_disliked_items WHERE user_id = " + str(
            session['user_id']), db.connection
    )

    if rec_engine == "random":
        query = "SELECT workout_id, RAND() as score FROM fbworkouts_meta ORDER BY score"
        results = pd.read_sql_query(query, db.connection)
        pred, scores = list(results.iloc[:, 0]), list(results.iloc[:, 1])
    elif rec_engine == "toppop":
        query = """
                SELECT workout_id, COUNT(workout_id) AS score
                FROM workout.user_item_interaction
                GROUP BY workout_id
                ORDER BY 2 DESC
                """
        results = pd.read_sql_query(query, db.connection)
        pred, scores = list(results.iloc[:, 0]), list(results.iloc[:, 1])
    else:
        uii = pd.read_sql_query(
            "SELECT * FROM user_item_interaction", db.connection)
        data = get_data(uii)

        if len(all_user_interactions) != 0:
            pred, scores = pred_i(data, session['user_id'])
        else:  # give random recommendations if user has no previous interactions
            query = "SELECT workout_id, RAND() as score FROM fbworkouts_meta ORDER BY score"
            results = pd.read_sql_query(query, db.connection)
            pred, scores = list(results.iloc[:, 0]), list(results.iloc[:, 1])

    # dct for predictions to scores
    pred_scores = {pred[i]: scores[i] for i in range(len(pred))}

    # get fbworkouts dataframe
    query = "SELECT * FROM fbworkouts"
    results = pd.read_sql_query(query, db.connection)

    # dictionary with keys as body focus and values as filtered list of workouts
    pred_dct = create_rec_lists(results, g.user, True)

    # dictionary with keys as body focus and values as dataframes with
    # fb_workouts_meta schema and rows sorted by scores
    rec_dct = {}
    for body_focus in pred_dct.keys():
        query = "SELECT * FROM fbworkouts_meta WHERE workout_id IN (" + str(
             pred_dct[body_focus])[1:-1] + ")"
        results = get_rec_sorted(pd.read_sql_query(
            query, db.connection), pred_scores)
        results['liked'] = results['workout_id'].apply(
            lambda x: x in list(all_user_interactions['workout_id']))
        results['disliked'] = results['workout_id'].apply(
            lambda x: x in list(user_disliked_items['workout_id']))

        rec_dct[body_focus.replace(
            '_', ' ').capitalize().replace('b', 'B')] = results[~results['disliked']]

    return render_template("recommendation_page.html", dropdown_option=rec_engine, rec_dct=rec_dct)


@app.route('/update', methods=['GET', 'POST'])
def update():
    form = WorkoutInformation()
    if form.validate_on_submit():  # update user table based on form inputs
        cur = db.connection.cursor()
        cur.execute(*update_preferences(form, g.user.user_id))
        db.connection.commit()
        cur.close()
        return redirect(url_for('recommendation_page'))

    # create dictionary from user series in order to prepopulate form with previous preferences
    user_dct = g.user[['equipment', 'training_type', 'min_duration', 'max_duration', 'min_calories',
                       'max_calories', 'min_difficulty', 'max_difficulty']].to_dict()
    for k, v in user_dct.items():
        if type(v) != str:
            user_dct[k] = int(v)
    return render_template('update_workout_info.html', form=form, user=user_dct)


@app.route('/history', methods=['GET', 'POST'])
def history_page():
    # if user is not logged in, redirect to login page
    if g.user is None:
        return redirect(url_for('login_page'))

    interaction_type = request.form.get("type")

    if interaction_type is None:
        return render_template("history_page.html", dropdown_option=None, rec_dct=None)

    # user's previous interactions
    all_user_interactions = pd.read_sql_query(
        "SELECT * FROM workout.user_item_interaction WHERE user_id = " + str(
            session['user_id']), db.connection
    )

    # user's disliked items
    user_disliked_items = pd.read_sql_query(
        "SELECT * FROM workout.user_disliked_items WHERE user_id = " + str(
            session['user_id']), db.connection
    )

    if interaction_type == "liked":
        user_interacted = all_user_interactions
    else:
        user_interacted = user_disliked_items

    # get fbworkouts dataframe
    query = "SELECT * FROM fbworkouts"
    results = pd.read_sql_query(query, db.connection)

    # dictionary with keys as body focus and values as list of liked/disliked workouts
    pred_dct = create_rec_lists(results, g.user, False)

    # dictionary with keys as body focus and values as dataframes with
    # fb_workouts_meta schema
    rec_dct = {}
    for body_focus in pred_dct.keys():
        if len(user_interacted) == 0:
            user_interacted_index = [999999]
        else:
            user_interacted_index = user_interacted['workout_id']

        query = "SELECT * FROM fbworkouts_meta WHERE workout_id IN (" + str(
            pred_dct[body_focus])[1:-1] + ") AND workout_id IN (" + str(list(user_interacted_index))[1:-1] + ")"
        results = pd.read_sql_query(query, db.connection)
        results['liked'] = results['workout_id'].apply(
            lambda x: x in list(all_user_interactions['workout_id']))
        results['disliked'] = results['workout_id'].apply(
            lambda x: x in list(user_disliked_items['workout_id']))

        rec_dct[body_focus.replace(
            '_', ' ').capitalize().replace('b', 'B')] = results
    
    return render_template("history_page.html", dropdown_option=interaction_type, rec_dct=rec_dct)


@app.route('/logout')
def logout():
    session.pop('user_id', None)  # removes session if currently in one
    return redirect(url_for('login_page'))


@app.route('/about')
def about_page():
    return render_template('about_page.html')


@app.route('/contact')
def contact_page():
    return render_template('contact_page.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/record_like/<user_id>/<workout_id>')
def record_like(user_id, workout_id):
    """
    Handler for like button event (record like)
    """

    all_user_interactions = pd.read_sql_query(
        "SELECT * FROM workout.user_item_interaction WHERE user_id = " +
        user_id + " and workout_id = " + workout_id, db.connection
    )

    cur = db.connection.cursor()
    if len(all_user_interactions) == 0:
        # no such interaction

        cur.execute(
            "INSERT INTO workout.user_item_interaction (user_id, workout_id) VALUES (%s, %s);", (int(user_id), int(workout_id)))
    else:
        pass

    cur.connection.commit()
    return user_id + " " + workout_id


@app.route('/remove_like/<user_id>/<workout_id>')
def remove_like(user_id, workout_id):
    """
    Handler for like button event (remove like)
    """

    all_user_interactions = pd.read_sql_query(
        "SELECT * FROM workout.user_item_interaction WHERE user_id = " +
        user_id + " and workout_id = " + workout_id, db.connection
    )

    cur = db.connection.cursor()
    if len(all_user_interactions) == 0:
        pass
    else:
        cur.execute(
            "DELETE FROM workout.user_item_interaction WHERE user_id = %s and workout_id = %s;", (int(user_id), int(workout_id)))

    cur.connection.commit()
    return user_id + " " + workout_id


@app.route('/record_dislike/<user_id>/<workout_id>')
def record_dislike(user_id, workout_id):
    """
    Handler for dislike button event (record like)
    """

    user_disliked_items = pd.read_sql_query(
        "SELECT * FROM workout.user_disliked_items WHERE user_id = " +
        user_id + " and workout_id = " + workout_id, db.connection
    )

    cur = db.connection.cursor()

    if len(user_disliked_items) == 0:
        # never disliked

        cur.execute(
            "INSERT INTO workout.user_disliked_items (user_id, workout_id) VALUES (%s, %s);", (int(user_id), int(workout_id)))
    else:
        pass

    cur.connection.commit()
    return user_id + " " + workout_id


@app.route('/remove_dislike/<user_id>/<workout_id>')
def remove_dislike(user_id, workout_id):
    """
    Handler for dislike button event (remove disliking)
    """

    user_disliked_items = pd.read_sql_query(
        "SELECT * FROM workout.user_disliked_items WHERE user_id = " +
        user_id + " and workout_id = " + workout_id, db.connection
    )

    cur = db.connection.cursor()
    if len(user_disliked_items) == 0:
        pass
    else:
        cur.execute(
            "DELETE FROM workout.user_disliked_items WHERE user_id = %s and workout_id = %s;", (int(user_id), int(workout_id)))

    cur.connection.commit()
    return user_id + " " + workout_id


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
