#import sys
#import getopt
import os
import rospy
from vrep_common.srv import *
from get_initial_object_belief import GetInitialObjectBelief, save_object_file
import time
import yaml
import numpy as np
import math

def get_pruned_saso_files(object_name, data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'SASOData_' + object_name in f and f.endswith('.txt') and '_24_' in f]
    files_prefix = list(set(["_".join(x.split('_')[0:-1]) + "_" for x in files]))
    return files_prefix;

def get_object_properties(object_id, object_property_dir, object_mesh_dir="g3db_meshes/"):
    object_properties_filename = (object_property_dir + "/" + object_id + '.yaml')
    if not os.path.exists(object_properties_filename):
        mesh_properties = get_default_object_properties(object_id)
    else:
        object_properties_filename = (object_property_dir + "/" + object_id + '.yaml')
        with open(object_properties_filename,'r') as stream:
            mesh_properties = yaml.load(stream)
    mesh_properties['mesh_dir'] = object_mesh_dir 
    return mesh_properties

def get_default_object_properties(object_id):
    if 'g3db' not in object_id:
        return get_object_properties_for_pure_shape(object_id)
    else:
        return {}

def get_object_properties_for_pure_shape(object_id_str):
    ans = {}
    ans['signal_name'] = 'pure_shape'
    ans['mesh_name'] = object_id_str.split('_')[0]
    object_id = int(object_id_str.split('_')[1])
    
    if(object_id <=10):
        object_id = object_id*10.0
    #ans['size_xyz'] = [object_id/100.0, object_id/100.0, 1.0] #scaling factor
    #ans['size_xyz'] = [i/0.10 for i in ans['size_xyz']]
    ans['actions'] = [{'set_size' : [object_id/100.0, object_id/100.0, 1.0]}, 'set_mass', 'move_to_table']
    
    return ans
def update_size_action(action, action_value):
    mp = {}
    update_object('get_size', mp)
    object_size = mp['object_size']
    print object_size
    action_ans = 'set_size'
    
    action_value_ans = [1.0 for x in object_size]

    j= None
    if(action.split('_')[-1] == 'x'):
        j = 0
    if(action.split('_')[-1] == 'y'):
        j = 1
    if(action.split('_')[-1] == 'z'):
        j = 2
        
    if('set_size_abs_xyz_' in action):
        val = action_value/object_size[j]
        action_value_ans = [val,val,val]
    elif('set_size_abs_xy_' in action):
        assert(j<2)
        val = action_value/object_size[j]
        action_value_ans = [val,val,1.0]
    elif('set_size_abs_' in action):
        val = action_value/object_size[j]
        action_value_ans[j] = val
    elif('rotate' in action):
        action_ans = 'rotate'
        action_value_ans = [0.0,0.0,0.0]
        action_value_ans[j] = action_value*3.14/180.0 #Api takes radians
    else:
        action_ans = action
        action_value_ans = action_value
        
    return (action_ans, action_value_ans)

def update_object(action_, mesh_properties):
    rospy.wait_for_service('/vrep/simRosCallScriptFunction')
    try:
        call_function = rospy.ServiceProxy('/vrep/simRosCallScriptFunction', simRosCallScriptFunction)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    if type(action_)==dict:
        action = action_.keys()[0]
        action_value = action_[action]
        if(action!='set_size'):
            action,action_value = update_size_action(action,action_value)
    else:
        action = action_
       
    if(action == 'get_collision_info'):
        try:
            #6 is for customization script
            resp1 = call_function("rosCollisionFunction@highTable", 1, [], [],  [], action)
            print resp1
            mesh_properties["collisions"] = resp1.outputInts[0]
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            
    if(action == 'get_size'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  [], action)
            mesh_properties["object_size"] = resp1.outputFloats
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action in [ 'set_size' , 'rotate']):
        print action_value
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], action_value,  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action == 'set_mass'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [0.3027],  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action in ['move_to_table', 'reorient_bounding_box'] ):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    
    if(action == 'set_object_pose'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], action_value,  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action == 'get_object_pose'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  [], action)
            mesh_properties["object_initial_pose_z"] = resp1.outputFloats[2]
            mesh_properties["object_pose"] = resp1.outputFloats
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    
    if(action == 'load_object'):
        mesh_path = mesh_properties['mesh_name']
        if('mesh_dir' in mesh_properties.keys()):
            mesh_path = os.path.abspath(mesh_properties['mesh_dir']) + "/" + mesh_properties['mesh_name']
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  
            [mesh_properties['signal_name'], mesh_path], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

def save_point_cloud(object_id, object_property_dir, object_mesh_dir, point_cloud_dir):
    mesh_properties = add_object_in_scene(object_id,object_property_dir, object_mesh_dir)
    start_stop_simulation('Start')
    assert('pick_point' in mesh_properties.keys())
    save_object_file(point_cloud_dir + "/" + object_id, False)
    start_stop_simulation('Stop')
    time.sleep(2)
    
def add_object_in_scene(object_id, object_property_dir, object_mesh_dir="g3db_meshes/"):
    
    mesh_properties = get_object_properties(object_id, object_property_dir, object_mesh_dir)
    update_object('load_object', mesh_properties)
    add_object_from_properties(mesh_properties)
    return mesh_properties
    
def add_object_from_properties(mesh_properties):
    
    #load_object(mesh_location, signal_name)
    print mesh_properties
    for action in mesh_properties['actions']:
        print action
        #a = raw_input("Proceed?")
        update_object(action, mesh_properties)
    
    if "object_initial_pose_z" not in mesh_properties.keys():
        update_object("get_object_pose", mesh_properties)
    if "object_min_z" not in mesh_properties.keys():
        mesh_properties["object_min_z"] = mesh_properties["object_initial_pose_z"] - 0.0048
    #if "object_size"
    print mesh_properties
    if 'pick_point' in mesh_properties.keys():
        object_pose = place_object_at_initial_location(mesh_properties)
        mesh_properties['final_initial_object_pose'] = object_pose
    return mesh_properties

def get_object_pick_point(object_id, object_property_dir):
    mesh_properties = add_object_in_scene(object_id, object_property_dir)
    get_object_characteristics(mesh_properties)
    
def get_object_pick_point(mesh_properties):
    object_pose = list(mesh_properties["object_pose"][0:3])
    hack_value_x = 0.07
    #Hack: Move in x to makesure full object is in camera view
    object_pose[0] = object_pose[0] + hack_value_x
    update_object({'set_object_pose': object_pose}, {})
    start_stop_simulation('Start')
    """
    Assuming no initial collision
    update_object('get_collision_info', mesh_properties)
    move_forward_count = 0
    while mesh_properties["collisions"] > 0:
        #move object by one cm along x axis
        object_pose = mesh_properties["object_pose"][0:3]
        object_pose[0] = object_pose[0] + 0.01
        update_object({'set_object_pose': object_pose}, mesh_properties)
        update_object("get_object_pose", mesh_properties)
        update_object('get_collision_info', mesh_properties)
        move_forward_count = move_forward_count + 1
        if (move_forward_count > 10):
            mesh_properties['collision_not_resolved'] = True
            break
    #start_stop_simulation('Stop')
    #update_object({'set_object_pose': object_pose}, mesh_properties)
    """
    #If collision is resolved, find the x,y position for grasp        
    if 'collision_not_resolved' not in mesh_properties.keys():
        #start simulation
        #start_stop_simulation('Start')
        mico_target_frame_pose = get_mico_target_frame_pose()
        min_z = mico_target_frame_pose.pose.pose.position.z -0.005;
        max_z = mico_target_frame_pose.pose.pose.position.z +0.005;
        #get object point cloud
        giob = GetInitialObjectBelief(None, True)
        pick_point = giob.get_nearest_pick_point(min_z, max_z)
        #rospy.signal_shutdown("Kinect listener not needed")
        start_stop_simulation('Stop')
        time.sleep(1)
        if(pick_point[0] is None):
            mesh_properties['object_pickable'] = False
        else:
            mesh_properties['object_pickable'] = True
            print pick_point
            #update_object({'set_object_pose': object_pose}, mesh_properties)
            set_grasp_point(pick_point)
            pick_point[0] = pick_point[0]- hack_value_x
            print pick_point
            mesh_properties['pick_point'] = pick_point
        

def check_for_object_collision(mesh_properties):
    object_pose = place_object_at_initial_location(mesh_properties)
    collision_detected = False
    for x in np.arange(-0.04,0,0.01):
        if collision_detected:
            break
        for y in np.arange(-0.04,0.05,0.01):
            start_stop_simulation('Start')
            update_object('get_collision_info', mesh_properties)
            start_stop_simulation('Stop')
            time.sleep(1)
            if mesh_properties["collisions"] > 0:
                collision_detected = True
                break
            object_pose_new = object_pose[0:3]
            object_pose_new[0] = object_pose[0] + x
            object_pose_new[1] = object_pose[1] + y
            update_object({'set_object_pose': object_pose_new}, {})
    mesh_properties['colliding_with_gripper'] = collision_detected
   
def check_for_object_stability(mesh_properties):
    object_pose = place_object_at_initial_location(mesh_properties)
    object_stable = True
    
    for x in range(0,19):
        start_stop_simulation('Start')
        for i in range(0,x): 
            move_gripper([0.01, 0.0,0.0])
        for i in range(0,8):
            move_gripper([0, 0.01,0.0])
        if has_object_fallen(mesh_properties):
            object_stable = False
            start_stop_simulation('Stop')
            time.sleep(1)
            break
        start_stop_simulation('Stop')
        time.sleep(1)
        start_stop_simulation('Start')
        for i in range(0, x): 
            move_gripper([0.01, 0.0,0.0])
        for i in range(0,7):
            move_gripper([0, -0.01,0.0])
        if has_object_fallen(mesh_properties):
            object_stable = False
            start_stop_simulation('Stop')
            time.sleep(1)
            break
        start_stop_simulation('Stop')
        time.sleep(1)
    mesh_properties['object_stable'] = object_stable

   
def move_gripper(move_pos):
    mico_target_pose = get_any_object_pose('Mico_target')
    current_pos = [0,0,0]
    current_pos[0]= mico_target_pose.pose.pose.position.x + move_pos[0]
    current_pos[1]= mico_target_pose.pose.pose.position.y + move_pos[1]
    current_pos[2]= mico_target_pose.pose.pose.position.z + move_pos[2]
    set_any_object_position('Mico_target', current_pos)
    time.sleep(1)
    
def place_object_at_initial_location(mesh_properties):
    
    object_pose = (list(mesh_properties["object_pose"]))[0:3]
    if('pick_point' in mesh_properties.keys()):
        IDEAL_PICK_POINT_X = object_pose[0] -0.03
        IDEAL_PICK_POINT_Y = object_pose[1]

        x_diff = IDEAL_PICK_POINT_X - mesh_properties['pick_point'][0]
        y_diff = IDEAL_PICK_POINT_Y - mesh_properties['pick_point'][1]

        object_pose[0] = object_pose[0] + x_diff
        object_pose[1] = object_pose[1] + y_diff
        update_object({'set_object_pose': object_pose}, {})
    return object_pose
  
def has_object_fallen(mesh_properties, use_quaternion = True):
    temp = {}
    update_object("get_object_pose", temp)
    object_pose = temp["object_pose"][0:3]
    
    if(use_quaternion):
        (X,Y,Z) = quaternion_to_euler_angle(temp["object_pose"][6], temp["object_pose"][3], temp["object_pose"][4], temp["object_pose"][5])
        print repr(X) + " " + repr(Y) + " " + repr(Z)
        return abs(X) > 45 or abs(Y) > 45
    else: #use height
        return (object_pose[2] < mesh_properties["object_min_z"])
        
    

    
def set_grasp_point(pick_point):
    set_any_object_position('GraspPoint', pick_point)

def get_mico_target_frame_pose():
    return get_any_object_pose('MicoTargetFrame')
    
def set_any_object_position(object_name, object_position):
    rospy.wait_for_service('/vrep/simRosGetObjectHandle')
    try:
        get_handle = rospy.ServiceProxy('/vrep/simRosGetObjectHandle', simRosGetObjectHandle)
        resp1 = get_handle(object_name)
        #return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        
    rospy.wait_for_service('/vrep/simRosCallScriptFunction')
    try:
        call_function = rospy.ServiceProxy('/vrep/simRosCallScriptFunction', simRosCallScriptFunction)
        resp1 = call_function('rosSetObjectPosition@TheAlmighty', 6, [resp1.handle], object_position,  [], '')
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        
def get_any_object_pose(object_name):
    rospy.wait_for_service('/vrep/simRosGetObjectHandle')
    try:
        get_handle = rospy.ServiceProxy('/vrep/simRosGetObjectHandle', simRosGetObjectHandle)
        resp1 = get_handle(object_name)
        #return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        
    rospy.wait_for_service('/vrep/simRosGetObjectPose')
    try:
        get_object_pose = rospy.ServiceProxy('/vrep/simRosGetObjectPose', simRosGetObjectPose)
        resp2 = get_object_pose(resp1.handle, -1)
        return resp2
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        
def start_stop_simulation(service_type = 'Start'):
    service_name = '/vrep/simRos' + service_type + 'Simulation'
    rospy.wait_for_service(service_name)
    try:
        if(service_type =='Start'):
            a = rospy.ServiceProxy(service_name, simRosStartSimulation)
        if(service_type =='Stop'):
            a = rospy.ServiceProxy(service_name, simRosStopSimulation)
        resp1 = a()
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z        
#If the object is already there, this function will first remove the object 
#deprecated use the update_object function instead
def load_object(object_file_name, signal_name = 'mesh_location'):
    rospy.wait_for_service('/vrep/simRosSetStringSignal')
    try:
        set_signal = rospy.ServiceProxy('/vrep/simRosSetStringSignal', simRosSetStringSignal)
        resp1 = set_signal(signal_name, os.path.abspath(object_file_name))
        #return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    """
    ros_command = 'rosservice call /vrep/simRosSetStringSignal "signalName: '
    ros_command = ros_command + "'mesh_location' \n signalValue: '"
    ros_command = ros_command + os.path.abspath(object_file_name) + "'" + '"'
    print ros_command
    os.system(ros_command)
    """
def remove_object():
    rospy.wait_for_service('/vrep/simRosGetObjectHandle')
    try:
        get_handle = rospy.ServiceProxy('/vrep/simRosGetObjectHandle', simRosGetObjectHandle)
        resp1 = get_handle('Cup')
        #return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    rospy.wait_for_service('/vrep/simRosRemoveObject')
    try:
        remove_obj = rospy.ServiceProxy('/vrep/simRosRemoveObject', simRosRemoveObject)
        remove_obj(resp1.handle)
        #return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    
    """
    ros_command = 'rosservice call /vrep/simRosGetObjectHandle "objectName: '
    ros_command = ros_command + "'Cup'" + '"'
    var = os.system(ros_command)
    ros_command = 'rosservice call /vrep/simRosRemoveObject "' + var + '"'
    os.system(ros_command)
    """
    
