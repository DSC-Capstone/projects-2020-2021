import pandas as pd

def create_rec_lists(workouts, user, filter):
    """
    Takes in dataframe of workouts (with fbworkouts schema) and users series
    (with users schema) and returns a dictionary with body focus as keys and
    workout ids in lists, with workouts not matching users' preferences filtered out
    if specified by filter parameter.
    """
    def training_type_helper(str):
        """
        Takes in a workout's training types and returns True if at least
        one matches user's preferred training types else False
        """
        training_type_list = str.split(', ')
        for t in training_type_list:
            if user[t] == 1:
                return True
        return False

    def equipment_helper(str):
        """
        Takes in workout's required equipment and returns True if
        user has ALL of the required equipment else False or if workout lists
        no equipment
        """
        equipment_list = str.split(', ')
        if 'no_equipment' in equipment_list:
            return True
        for e in equipment_list:
            if e == 'no_equipment':
                pass
            elif user[e] == 0:
                return False
        return True

    def calorie_helper(series):
        """
        Takes in workout's series , and returns True if user's preferred min/max
        calorie range has some overlap with the workout's calorie range
        """
        if user['max_calories'] < series['min_calorie_burn'] or series['max_calorie_burn'] < user['min_calories']:
            return False
        return True

    def in_range_helper(x, attr):
        """
        Takes in workout's attr (difficulty or duration) and returns True if it is
        within the range of user's preferred attr range (inclusive)
        """
        if x >= user['min_' + attr] and x <= user['max_' + attr]:
            return True
        return False

    if filter:
        # filter
        workouts = workouts[workouts['duration'].apply(in_range_helper, args=('duration',))]
        workouts = workouts[workouts['difficulty'].apply(in_range_helper, args=('difficulty',))]
        workouts = workouts[workouts['equipment'].apply(equipment_helper)]
        workouts = workouts[workouts['training_type'].apply(training_type_helper)]
        workouts = workouts[workouts.apply(calorie_helper,axis=1)]

    def get_body_focus(body_focus):
        """
        Helper function that takes in a body focus and returns list of workout
        ids with that body focus
        """
        return list(workouts.loc[workouts[body_focus]==1,'workout_id'])

    dct = {x:get_body_focus(x) for x in ['upper_body','lower_body','core','total_body']}
    return dct


def get_rec_sorted(workouts_meta, pred_scores):
    """
    Helper function that takes in a dataframe (with workouts_meta schema) and
    a dictionary with (key, value) as (workout id, score) and returns the
    dataframe sorted by model scores
    """
    workouts_meta['score'] = workouts_meta['workout_id'].apply(lambda x:pred_scores[x])
    sorted = workouts_meta.sort_values('score',ascending=False)
    return sorted
