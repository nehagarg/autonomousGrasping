
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
    elif type in ['all_cylinders', 'cylinders_train', 'cylinders_test']:
        a = ['9','8','7','75','85']
        if 'train' in type:
            a = ['9','8','7']
        if 'test' in type:
            a = ['75', '85']
        pattern_list = ['Cylinder_' + i for i in a]
    elif type=='g3db_instances':
        pattern_list = get_g3db_instances()
    elif type=='cylinder_and_g3db_instances':
        pattern_list = get_grasping_object_name_list('all_cylinders')
        pattern_list = pattern_list + get_grasping_object_name_list('g3db_instances')
    elif type=='79_toy_dog_final':
        pattern_list_ = get_g3db_instances()
        pattern_list = [x  for x in pattern_list_  if '79_toy_dog_final' in x]
    elif type=='objects_modified':
        pattern_list = ['5_bottle_final-15-Dec-2015-15-43-28_instance0']
	pattern_list.append('62_mouse_final-21-Nov-2015-06-46-41_instance0') 
	pattern_list.append('6_jar_final-14-Nov-2015-19-14-33_instance0')
        pattern_list.append('6_jar_final-20-Dec-2015-09-26-23_instance0')

    else:
        pattern_list=[type]
        
    return pattern_list

def get_g3db_instances():
    try:
        import rospkg
        rospack = rospkg.RosPack()
        grasping_ros_mico_path = rospack.get_path('grasping_ros_mico')
    except:
        #give ncl absolute path
        grasping_ros_mico_path = '/users/ngarg211/WORK_FOLDER/neha_github/autonomousGrasping/grasping_ros_mico'
    g3db_object_list_file = grasping_ros_mico_path + "/g3db_object_labels/object_instances/object_instances_updated/object_instance_names.txt"
    ans = []
    with open(g3db_object_list_file, 'r') as f:
        lines = f.readlines()
        ans = [l.strip() for l in lines]
    return ans