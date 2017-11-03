import sys
import getopt
import os
import rospy
from vrep_common.srv import *

def load_object(object_file_name):
    rospy.wait_for_service('/vrep/simRosSetStringSignal')
    try:
        set_signal = rospy.ServiceProxy('/vrep/simRosSetStringSignal', simRosSetStringSignal)
        resp1 = set_signal('mesh_location', os.path.abspath(object_file_name))
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
    
def label_object(object_file_name, dir_name):
    if(os.path.isdir(object_file_name)):
        files = [os.path.join(object_file_name, f) for f in os.listdir(object_file_name) if '.obj' in f]
        for file in files:
            label_object(file, dir_name)
    print "Labeling " + os.path.basename(object_file_name)
    output_file_name = os.path.basename(object_file_name).replace('.obj', '.label')
    if (os.path.exists(os.path.join(dir_name, output_file_name))):
       ans =  raw_input("Label exists. Relabel?")
       if ans=='n':
           return
    load_object(object_file_name)
    label = 'ST'
    label_in = raw_input("Please provide label: S|T|ST:")
    if label_in:
        label = label_in
    
    with open(os.path.join(dir_name, output_file_name), 'w') as f:
        f.write(os.path.basename(object_file_name) + ' ' + label )
    remove_object()
    
def main():
    dir_name = './'
    opts, args = getopt.getopt(sys.argv[1:],"ho:",["outdir=",])
    
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'label_g3db_objects.py -o output_dir_name <object_file_name>'
         sys.exit()
      elif opt in ("-o", "--outdir"):
         dir_name = arg
    object_file_name = args[0]
    label_object(object_file_name, dir_name)

if __name__ == '__main__':
    main()    