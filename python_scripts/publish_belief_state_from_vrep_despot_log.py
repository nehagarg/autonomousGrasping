import roslib; roslib.load_manifest('grasping_ros_mico')
from grasping_ros_mico.msg import Belief
from grasping_ros_mico.msg import State
from geometry_msgs.msg import PoseStamped
import rospy
from log_file_parser import ParseLogFile
import sys



def getGripperState(state_value):

    degree_readings = []
    degree_readings.append(state_value.fj1*180/3.14)
    degree_readings.append(state_value.fj2*180/3.14)
    degree_readings.append(state_value.fj3*180/3.14)
    degree_readings.append(state_value.fj4*180/3.14)
    
    if (degree_readings[0] > 22 ) and  (degree_readings[1] > 85) and (degree_readings[2] > 22) and (degree_readings[3] > 85):
    #joint1 > 20 joint2 > 85 
            return 1
    
   
    if (degree_readings[1] > 25) and (degree_readings[3] > 25):  #Changed from 45 to 25 looking at data
        #joint1 > 2 joint2 > 45 return 2
        return 2;
    
    return 0;

def getPublishableBelief(belief_array):
    belief_msg = Belief()
    belief_msg.numPars = len(belief_array)
    for i in range(0,len(belief_array)):
        belief_msg.belief.append(belief_array[i]['state'].o_x);
        belief_msg.belief.append(belief_array[i]['state'].o_y);
        belief_msg.belief.append(float(belief_array[i]['weight']));
    
    return belief_msg
         
           
def getPublishableState(state_value, touch_0, touch_1):
    msg = State()
    msg.gripper_pose = PoseStamped();
    msg.gripper_pose.pose.position.x = state_value.g_x
    msg.gripper_pose.pose.position.y = state_value.g_y
    msg.object_pose = PoseStamped()
    msg.object_pose.pose.position.x = state_value.o_x
    msg.object_pose.pose.position.y = state_value.o_y
    
    gripper_status = getGripperState(state_value)
    if(gripper_status == 1):
        msg.observation = 5 #OBS_NSTABLE;
    elif(gripper_status == 2):
        msg.observation = 4 #OBS_STABLE;
    else:
        msg.observation = touch_0 * 2 + touch_1
    
    return msg
 
if __name__ == '__main__':
    
    log_filename = sys.argv[1]
    
    
    
    
    pub_gripper = rospy.Publisher('gripper_pose', State, queue_size=10);
    pub_belief = rospy.Publisher("object_pose", Belief, queue_size= 10);
    rospy.init_node('python_belief_state_publisher')
    rate = rospy.Rate(10000)
    rate.sleep()
    
    lfp =  ParseLogFile(log_filename, 'vrep', 0, 'vrep')
    for i in range(0, len(lfp.stepInfo_)):
        if i==0:
            state_value = lfp.roundInfo_['state']
            touch_0 = 0
            touch_1 = 0
        else:
            state_value = lfp.stepInfo_[i-1]['state']
            touch_0 = lfp.stepInfo_[i-1]['obs'].sensor_obs[0]
            touch_1 = lfp.stepInfo_[i-1]['obs'].sensor_obs[1]
            
        p_State = getPublishableState(state_value, touch_0, touch_1)
        p_Belief = getPublishableBelief(lfp.stepInfo_[i]['belief'])
        pub_gripper.publish(p_State)
        pub_belief.publish(p_Belief)
        rate.sleep()
        name = raw_input("Continue with next step " + repr(i) + " " + lfp.stepInfo_[i]['action'] ) 
            
    
    
    