def return_breathing_flow_rate(breathing_flow_rate):
    # Converting from activity type to physical parameters for breathing_flow_rate
    if breathing_flow_rate == 'resting':
        breathing_flow_rate = 0.4927131 #0.29 ft3/min
    elif breathing_flow_rate == 'moderate_exercise':
        breathing_flow_rate = 2.3446349 #1.38 ft3/min
    elif breathing_flow_rate == 'light_exercise':
        breathing_flow_rate = 1.3761987 #0.8 ft3/min
    else:
        breathing_flow_rate = 0
     
    return breathing_flow_rate


def return_air_exchange_rate(air_exchange_rate):
    # Converting from activity type to physical parameters for air_exchange_rate
    if air_exchange_rate == 'open_windows':
        air_exchange_rate = 2  #hr-1
    elif air_exchange_rate == 'closed_windows':
        air_exchange_rate = 0.3 #hr-1
    elif air_exchange_rate == 'mechanical_ventilation':
        air_exchange_rate = 3  #hr-1
    elif air_exchange_rate == 'fans':
        air_exchange_rate = 6  #hr-1
    elif air_exchange_rate == 'advanced_mechanical_ventilation':
        air_exchange_rate = 8  #hr-1

    return air_exchange_rate 


def return_mask_passage_prob(mask_passage_prob):
    # Converting from activity type to physical parameters for mask_passage_prob

    if mask_passage_prob == None:
        mask_passage_prob = 1
    elif mask_passage_prob == 'Cotton':
        mask_passage_prob = 0.5
    elif mask_passage_prob == 'Multilayer':
        mask_passage_prob = 0.3
    elif mask_passage_prob == 'Surgical':
        mask_passage_prob = 0.1
    elif mask_passage_prob == 'N95':
        mask_passage_prob = 0.05

    return mask_passage_prob


def return_exhaled_air_inf(exhaled_air_inf):
    # Converting from activity type to physical parameters for Exhaled air infectivity
    if exhaled_air_inf == 'talking_whisper':
        exhaled_air_inf = 28.9580743587 #q/m3
    
    elif exhaled_air_inf == 'talking_loud':
        exhaled_air_inf = 141.965193807 #q/m3

    elif exhaled_air_inf == 'breathing_heavy':
        exhaled_air_inf = 8.82868120692 #q/m3

    elif exhaled_air_inf == 'talking_normal':
        exhaled_air_inf = 72.0420386485 #q/m3
    else:
        exhaled_air_inf = 0
        
    return exhaled_air_inf