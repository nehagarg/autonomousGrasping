import sys
import getopt
import os
import rospy
from vrep_common.srv import *

import yaml

def get_object_properties(object_id, object_property_dir):
    if 'G3DB' not in object_id:
        mesh_properties = get_object_properties_for_pure_shape(object_id)
    else:
        object_properties_filename = (object_property_dir + "/" + object_id + '.yaml')
        with open(object_properties_filename,'r') as stream:
            mesh_properties = yaml.load(stream)
        mesh_properties['signal_name'] = 'mesh_location'
        mesh_properties['mesh_name'] = "g3db_meshes" + mesh_properties['mesh_name']
    return mesh_properties
    
def get_object_properties_for_pure_shape(object_id_str):
    ans = {}
    ans['signal_name'] = 'pure_shape'
    ans['mesh_name'] = object_id_str.split('_')[0]
    object_id = int(object_id_str.split('_')[1])
    
    if(object_id <=10):
        object_id = object_id*10.0
    ans['size_xyz'] = [object_id/100.0, object_id/100.0, 1.0] #scaling factor
    #ans['size_xyz'] = [i/0.10 for i in ans['size_xyz']]
    ans['actions'] = ['set_size', 'set_mass', 'move_to_table']
    
    return ans

def update_object(action, mesh_properties):
    rospy.wait_for_service('/vrep/simRosCallScriptFunction')
    try:
        call_function = rospy.ServiceProxy('/vrep/simRosCallScriptFunction', simRosCallScriptFunction)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    
    if(action == 'set_size'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], mesh_properties['size_xyz'],  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action == 'set_mass'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [0.3027],  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action == 'move_to_table'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  [], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if(action == 'get_object_pose'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  [], action)
            mesh_properties["object_initial_pose_z"] = resp1.outputFloats[2]
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    
    if(action == 'load_object'):
        try:
            #6 is for customization script
            resp1 = call_function('rosUpdateObject@TheAlmighty', 6, [], [],  
            [mesh_properties['signal_name'], mesh_properties['mesh_name']], action)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
   
def add_object_in_scene(object_id, object_property_dir):
    
    mesh_properties = get_object_properties(object_id, object_property_dir)
    
    update_object('load_object', mesh_properties)
    #load_object(mesh_location, signal_name)
    
    for action in mesh_properties['actions']:
        print action
        update_object(action, mesh_properties)
    
    if "object_initial_pose_z" not in mesh_properties.keys():
        update_object("get_object_pose", mesh_properties)
    if "object_min_z" not in mesh_properties.keys():
        mesh_properties["object_min_z"] = mesh_properties["object_initial_pose_z"] - 0.0048
    
    return mesh_properties
    
#If the object is already there, this function will first remove the object 
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
    
