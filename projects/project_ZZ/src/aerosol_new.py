
def return_aerosol_transmission_rate(floor_area, room_height,
                            air_exchange_rate, aerosol_filtration_eff, relative_humidity, 
                            breathing_flow_rate, exhaled_air_inf, mask_passage_prob, 
                            max_viral_deact_rate=0.3, max_aerosol_radius=2, primary_outdoor_air_fraction=0.2):

        
    mean_ceiling_height_m = room_height * 0.3048 #m3
    room_vol = floor_area * room_height  # ft3
    room_vol_m = 0.0283168 * room_vol  # m3

    fresh_rate = room_vol * air_exchange_rate / 60  # ft3/min
    recirc_rate = fresh_rate * (1/primary_outdoor_air_fraction - 1)  # ft3/min
    air_filt_rate = aerosol_filtration_eff * recirc_rate * 60 / room_vol  # /hr
    eff_aerosol_radius = ((0.4 / (1 - relative_humidity)) ** (1 / 3)) * max_aerosol_radius
    viral_deact_rate = max_viral_deact_rate * relative_humidity
    sett_speed = 3 * (eff_aerosol_radius / 5) ** 2  # mm/s
    sett_speed = sett_speed * 60 * 60 / 1000  # m/hr
    conc_relax_rate = air_exchange_rate + air_filt_rate + viral_deact_rate + sett_speed / mean_ceiling_height_m  # /hr
    airb_trans_rate = ((breathing_flow_rate * mask_passage_prob) ** 2) * exhaled_air_inf / (room_vol_m * conc_relax_rate)
    
    return airb_trans_rate #This is mean number of transmissions per hour between a single pair of infected / healthy individuals


