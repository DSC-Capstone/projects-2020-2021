from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectMultipleField, IntegerField, ValidationError, widgets, BooleanField
from wtforms.validators import DataRequired, Email, EqualTo, NumberRange

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()

class WorkoutInformation(FlaskForm):
    equipment = MultiCheckboxField('My Available Equipment', choices=[
        ('barbell','Barbell'), ('bench','Bench'), ('dumbbell','Dumbbell'),
        ('exercise_band','Exercise Band'), ('jump_rope','Jump Rope') ,
        ('kettlebell','Kettlebell'), ('mat','Mat'), ('medicine_ball','Medicine Ball'),
        ('physioball','Physioball'), #('no_equipment', 'No Equipment')
        ('sandbag','Sandbag'), ('stationary_bike','Stationary Bike')
    ])
    no_equipment = BooleanField('I have no equipment', render_kw={'onchange': "hideOptions(equipment_div)"})

    training_type = MultiCheckboxField('My Preferred Training Types', choices=[
        ('barre', 'Barre'), ('balance_agility', 'Balance Agility'), ('cardiovascular',
        'Cardiovascular'), ('hiit', 'HIIT'), ('low_impact', 'Low Impact'),
        ('pilates', 'Pilates'), ('plyometric', 'Plyometric'), ('strength_training',
        'Strength Training'), ('stretching_flexibility','Stretching/Flexibility'),
        ('toning','Toning'), ('warm_up_cool_down','Warm Up/Cool Down'),
        ('aerobics_step','Aerobics/Step')
    ])
    no_training_type = BooleanField('I have no preferred training types', render_kw={'onchange': "hideOptions(training_type_div)"})

    min_duration = IntegerField('Minimum Duration',
                    validators=[DataRequired(), NumberRange(1,120)]) # fb workout between 3-96 minutes
    max_duration = IntegerField('Maximum Duration',
                    validators=[DataRequired(), NumberRange(1,120)])

    min_calories = IntegerField('Minimum Calories',
                    validators=[DataRequired(), NumberRange(1,1300)]) # fb workouts between 12-260 calori burn
    max_calories = IntegerField('Maximum Calories',
                    validators=[DataRequired(), NumberRange(1,1300)])

    min_difficulty = IntegerField('Minimum Difficulty',
                    validators=[DataRequired(), NumberRange(1,5)])
    max_difficulty = IntegerField('Maximum Difficulty',
                    validators=[DataRequired(), NumberRange(1,5)])

    submit = SubmitField('Register')

    def validate_max_duration(form, field):
        if field.data < form.min_duration.data:
            raise ValidationError('Minimum Duration must be less than Maximum Duration')

    def validate_max_calories(form, field):
        if field.data < form.min_calories.data:
            raise ValidationError('Minimum Calories must be less than Maximum Calories')

    def validate_max_difficulty(form, field):
        if field.data < form.min_difficulty.data:
            raise ValidationError('Minimum Difficulty must be less than Maximum Difficulty')

    def validate_no_equipment(form, field):
        if field.data == False and form.equipment.data == []:
            raise ValidationError('Must Select Available Equipment or No Equipment Option')

    def validate_no_training_type(form, field):
        if field.data == False and form.training_type.data == []:
            raise ValidationError('Must SelectPreferred Training Type or No Preferred Training Type Option')

class RegistrationForm(WorkoutInformation):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                validators=[DataRequired(), EqualTo('password')])
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
