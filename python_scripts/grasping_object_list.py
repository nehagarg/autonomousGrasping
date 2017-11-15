def get_grasping_object_name_list(type='used'):
    pattern_list = ['G3DB11_cheese_final-10-mar-2016']
    pattern_list.append('G3DB39_beerbottle_final-11-mar-2016')
    pattern_list.append('G3DB40_carafe_final-11-mar-2016')
    pattern_list.append('G3DB41_jar_and_lid_final-11-mar-2016')
    pattern_list.append('G3DB43_wineglass2_final')
    pattern_list.append('G3DB48_bottle_and_plug_final')
    pattern_list.append('G3DB50_carton_final')
    pattern_list.append('G3DB54_candlestick_final')
    pattern_list.append('G3DB5_bottle_final-19-jan-2016')
    pattern_list.append('G3DB65_coffeemaker_final')
    pattern_list.append('G3DB66_chocolatebox_final')
    pattern_list.append('G3DB67_jug_final')
    pattern_list.append('G3DB73_juicebottle_final')
    pattern_list.append('G3DB84_yogurtcup_final')
    pattern_list.append('G3DB91_peppershaker_final')
    pattern_list.append('G3DB94_weight_final')
    assert(len(pattern_list)==16)
    if type=='used':
        pattern_list.remove('G3DB11_cheese_final-10-mar-2016') #cheese falls at initial state creation
    elif type=='coffee_yogurt_cup':
        pattern_list = ['G3DB1_Coffeecup_final-20-dec-2015']
        pattern_list.append('G3DB84_yogurtcup_final')
    elif type=='all_g3db':
        pattern_list.append('G3DB1_Coffeecup_final-20-dec-2015')
    elif type=='all_cylinders':
        a = ['9','8','7','75','85']
        pattern_list = ['Cylinder_' + i for i in a]
    else:
        pattern_list=[type]
    return pattern_list