
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
    elif type=='cylinder_and_g3db_instances_version7':
        pattern_list = get_grasping_object_name_list('all_cylinders')
        pattern_list = pattern_list + get_grasping_object_name_list('g3db_instances_version7')
    elif 'g3db_instances_' in type:
        pattern_list = get_g3db_instances(type.replace("g3db_instances_", ""))
    elif type=='79_toy_dog_final':
        pattern_list_ = get_g3db_instances()
        pattern_list = [x  for x in pattern_list_  if '79_toy_dog_final' in x]
    elif type=='objects_modified':
        pattern_list = ['5_bottle_final-15-Dec-2015-15-43-28_instance0']
	pattern_list.append('62_mouse_final-21-Nov-2015-06-46-41_instance0') 
	pattern_list.append('6_jar_final-14-Nov-2015-19-14-33_instance0')
        pattern_list.append('6_jar_final-20-Dec-2015-09-26-23_instance0')
    elif type=='training_towelstand':
        pattern_list = ['44_towelstand_final-16-Mar-2016-13-30-01_instance0']
        pattern_list.append('44_towelstand_final-15-Mar-2016-15-53-08_instance0')
        pattern_list.append('44_towelstand_final-15-Mar-2016-15-37-58_instance0')
    elif type=='g3db_train1_version7_without_towel':
        pattern_list = get_grasping_object_name_list('g3db_instances_train1_version7')
        pattern_list2 = get_grasping_object_name_list('training_towelstand')
        for a in pattern_list2:
            pattern_list.remove(a)
        
    else:
        pattern_list=[type]
        
    return pattern_list

def get_g3db_txt_file_path(server_name = 'ncl'):
    try:
        import rospkg
        rospack = rospkg.RosPack()
        grasping_ros_mico_path = rospack.get_path('grasping_ros_mico')
    except:
        #give ncl absolute path
        if(server_name == 'unicorn'):
            grasping_ros_mico_path = '/data/neha/WORK_FOLDER/neha_github/autonomousGrasping/grasping_ros_mico'
        else:
            grasping_ros_mico_path = '/users/ngarg211/WORK_FOLDER/neha_github/autonomousGrasping/grasping_ros_mico'
    g3db_object_list_file = grasping_ros_mico_path + "/g3db_object_labels/object_instances/object_instances_updated/"
    return g3db_object_list_file

def get_g3db_instances(type = 'all'):
    g3db_object_list_file = get_g3db_txt_file_path()
    if(type == 'all'):
       g3db_object_list_file = g3db_object_list_file +  "object_instance_names.txt"
    elif(type == 'for_classification'):
       g3db_object_list_file = g3db_object_list_file.replace('g3db_object_labels', 'g3db_object_labels_for_classification') +  "object_instance_names.txt" 
    else:
        g3db_object_list_file = g3db_object_list_file +  "object_instance_names_" + type + ".txt"
    
    ans = []
    with open(g3db_object_list_file, 'r') as f:
        lines = f.readlines()
        ans = [l.strip() for l in lines]
    return ans

def get_classes_from_g3db_instances(g3db_instances):
    g3db_classes = sorted(list(set([x.split('-')[0] for x in g3db_instances])))
    return g3db_classes
    
    
def classify_g3db_instances():
    ans = {}
    g3db_object_cylindrical = []
    g3db_object_square = []
    g3db_object_handles = []
    g3db_object_wine = []
    g3db_object_stands = []
    g3db_object_misc = []
    g3db_object_square.append('104_toaster_final')
    g3db_object_cylindrical.append('106_urn_final')
    g3db_object_misc.append('109_crab_final')
    g3db_object_misc.append('11_cheese_final')
    g3db_object_square.append('18_garbage_box_final')
    g3db_object_handles.append('1_Coffeecup_final')
    g3db_object_cylindrical.append('24_bowl')
    g3db_object_handles.append('24_mug_final')
    g3db_object_cylindrical.append('25_mug')
    g3db_object_misc.append('28_Spatula_final')
    g3db_object_cylindrical.append('30_fruit_juicer_final')
    g3db_object_cylindrical.append('39_beerbottle_final')
    g3db_object_stands.append('40_carafe_final')
    g3db_object_cylindrical.append('41_jar_and_lid_final')
    g3db_object_wine.append('42_wineglass_final')
    g3db_object_wine.append('43_wineglass2_final')
    g3db_object_stands.append('44_towelstand_final')
    g3db_object_cylindrical.append('48_bottle_and_plug_final')
    g3db_object_cylindrical.append('49_apple_final')
    g3db_object_square.append('50_carton_final')
    g3db_object_cylindrical.append('52_jar2_final')
    g3db_object_misc.append('54_candlestick_final')
    g3db_object_misc.append('56_headphones_final')
    g3db_object_cylindrical.append('5_bottle_final')
    g3db_object_misc.append('62_mouse_final')
    g3db_object_handles.append('63_candle_final')
    g3db_object_cylindrical.append('65_coffeemaker_final')
    g3db_object_square.append('66_chocolatebox_final')
    g3db_object_cylindrical.append('67_jug_final')
    g3db_object_cylindrical.append('6_jar_final')
    g3db_object_square.append('73_juicebottle_final')
    g3db_object_square.append('74_lamp_final')
    g3db_object_misc.append('75_vase3_final')
    g3db_object_wine.append('76_mirror_final')
    g3db_object_misc.append('77_napkinholder_final')
    g3db_object_misc.append('79_toy_dog_final')
    g3db_object_cylindrical.append('80_tincan_final')
    g3db_object_cylindrical.append('84_yogurtcup_final')
    g3db_object_square.append('86_bread_final')
    g3db_object_cylindrical.append('91_peppershaker_final')
    g3db_object_misc.append('92_shell_final')
    g3db_object_cylindrical.append('94_weight_final')
    ans['cylindrical'] =g3db_object_cylindrical 
    ans['square'] =g3db_object_square
    ans['handles'] =g3db_object_handles 
    ans['wine'] =g3db_object_wine
    ans['stands'] =g3db_object_stands 
    ans['misc'] =g3db_object_misc
    return ans

def get_object_classification(non_test_object_classes):
    g3db_object_classification = classify_g3db_instances()
    ans = {}
    for object_class_category in g3db_object_classification.keys():
        ans[object_class_category] = []
        for object_class in g3db_object_classification[object_class_category]:
            if object_class in non_test_object_classes:
                ans[object_class_category].append(object_class)
    return ans
def test_get_object_classification():
    g3db_instances = get_g3db_instances()
    g3db_instance_classes = get_classes_from_g3db_instances(g3db_instances)
    g3db_object_classification = get_object_classification(g3db_instance_classes)
    g3db_instances_classified = []
    for object_class_category in g3db_object_classification.keys():
        g3db_instances_classified = g3db_instances_classified + g3db_object_classification[object_class_category]
    print len(g3db_instances_classified)
    print len(g3db_instance_classes)
    assert(len(g3db_instances_classified) == len(g3db_instance_classes))
  
def sample_object_instances(num_samples, object_instance_list):
    import random
    a = range(0,len(object_instance_list))
    random.shuffle(a)
    ans=[]
    for i in range(0,num_samples):
        ans.append(object_instance_list[a[i]])
    return ans

def get_instances_for_class(object_class_list, type):
    object_instances = get_g3db_instances(type)
    target_object_instances = []
    for object_class in object_class_list:
        target_object_instances = target_object_instances + [x  for x in object_instances  if object_class in x]
    return target_object_instances

def sample_instances_from_class(num_samples, object_class_list, type):
    target_object_instances = get_instances_for_class(object_class_list, type)
    ans = sample_object_instances(num_samples, target_object_instances)
    return ans
    
def sample_one_from_class(num_samples, object_class_list, type): 
    ans = []
    square_classes = sample_object_instances(num_samples, object_class_list)
    for square_class in square_classes:
        ans = ans + sample_instances_from_class(1, [square_class], type)
    return ans

def sample_from_non_test_g3db_instances(update_file = False):
    non_test_object_instances = get_g3db_instances('non_test')
    non_test_object_classes = get_classes_from_g3db_instances(non_test_object_instances)
    #print non_test_object_classes
    non_test_object_classification = get_object_classification(non_test_object_classes)
    #print non_test_object_classification
    ans = []
    ans = ans + sample_one_from_class(4, non_test_object_classification['cylindrical'], 'non_test')
    ans = ans + sample_one_from_class(2, non_test_object_classification['square'], 'non_test')
    ans = ans + sample_instances_from_class(2, non_test_object_classification['handles'], 'non_test')
    ans = ans + sample_instances_from_class(1, non_test_object_classification['wine'], 'non_test')
    ans = ans + sample_instances_from_class(2, non_test_object_classification['stands'], 'non_test')
    ans = ans + sample_one_from_class(4, non_test_object_classification['misc'], 'non_test')
    
    print ans
    validation_set = sorted(list(set(non_test_object_instances) - set(ans)))
    if update_file:
        g3db_object_list_file = get_g3db_txt_file_path()
        g3db_train_file_name = g3db_object_list_file +  "object_instance_names_" + "train1" + ".txt"
        with open(g3db_train_file_name, 'w') as f:
            f.write("\n".join(ans))
            f.write("\n")
        g3db_validation_file_name = g3db_object_list_file +  "object_instance_names_" + "validation1" + ".txt"
        with open(g3db_validation_file_name, 'w') as f:
            f.write("\n".join(validation_set))
            f.write("\n")
    
    
    