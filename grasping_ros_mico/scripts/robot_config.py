#import rospy
from geometry_msgs.msg import PoseStamped


DEFAULT_FINGER_MAX_TURN = 6800
DEFAULT_FINGER_MAX_DIST = 18.9/2/1000
DEFAULT_HOME_POS = [0.212322831154, -0.257197618484, 0.509646713734, 1.63771402836, 1.11316478252, 0.134094119072] # default home in unit mq
# DEFAULT_ROBOT_TYPE = 'm1n6a200'
DEFAULT_ROBOT_TYPE = 'm1n6s200'

### New Global Variables
HOME_JOINT = [275.18, 175.20, 78.915, 242.86, 84.27, 75.136, 0]

HOME_POSE_R = PoseStamped()
HOME_POSE_T = PoseStamped()
PRE_PUSH = PoseStamped()
PRE_GRASP = PoseStamped()
PRE_PUSH_TRANS = PoseStamped()
PRE_PUSH_TRANS_2 = PoseStamped()
PRE_GRASP_TRANS1 = PoseStamped()
PRE_GRASP_TRANS = PoseStamped()
WALLET_PRE_GRASP = PoseStamped()
PRE_GRASP_2 = PoseStamped()
TOP_OF_BOOKS = PoseStamped()

PRE_GRASP_2.header.frame_id = 'm1n6s200_link_base'
PRE_GRASP_2.pose.position.x= 0.337906807661
PRE_GRASP_2.pose.position.y= -0.10480992496
PRE_GRASP_2.pose.position.z= 0.315148193836
PRE_GRASP_2.pose.orientation.x= -0.489324437899
PRE_GRASP_2.pose.orientation.y= -0.521763124658
PRE_GRASP_2.pose.orientation.z= -0.513538916863
PRE_GRASP_2.pose.orientation.w= -0.473922585545


TOP_OF_BOOKS.header.frame_id = 'm1n6s200_link_base'
TOP_OF_BOOKS.pose.position.x= -0.291403710842
TOP_OF_BOOKS.pose.position.y= 0.0422490760684
TOP_OF_BOOKS.pose.position.z= 0.403839349747
TOP_OF_BOOKS.pose.orientation.x= 0.58360194393
TOP_OF_BOOKS.pose.orientation.y= -0.428733914599
TOP_OF_BOOKS.pose.orientation.z= -0.332832332378
TOP_OF_BOOKS.pose.orientation.w= 0.604002185458


# combine push to rotate and translate
rot_pose = PoseStamped()
tra_pose = PoseStamped()
drill_pose = PoseStamped()

drill_pose.header.frame_id = 'm1n6s200_link_base'
drill_pose.pose.position.x = 0.275101393461 
drill_pose.pose.position.y = -0.186394318938
drill_pose.pose.position.z = 0.10991229862
drill_pose.pose.orientation.x = 0.572439432144
drill_pose.pose.orientation.y = -0.719276666641
drill_pose.pose.orientation.z = 0.310120344162
drill_pose.pose.orientation.w = 0.242444932461

rot_pose.header.frame_id = 'm1n6s200_link_base'
rot_pose.pose.position.x = -0.193705901504 
rot_pose.pose.position.y = -0.0820705890656
rot_pose.pose.position.z = 0.180212914944
rot_pose.pose.orientation.x = 0.477668315172 
rot_pose.pose.orientation.y = -0.543274521828
rot_pose.pose.orientation.z = 0.456881612539
rot_pose.pose.orientation.w = 0.517634034157 
#rot_pose.pose.position.x = 0.15342742204 
#rot_pose.pose.position.y = -0.134502753615
#rot_pose.pose.position.z = 0.1650252
#rot_pose.pose.orientation.x = 0.33012676239
#rot_pose.pose.orientation.y = -0.65131127834
#rot_pose.pose.orientation.z = 0.44107803
#rot_pose.pose.orientation.w = 0.5217854976

tra_pose.header.frame_id = 'm1n6s200_link_base'
tra_pose.pose.position.x = 0.128234450221 
tra_pose.pose.position.y = -0.288208603859
tra_pose.pose.position.z = 0.1101522147
tra_pose.pose.orientation.x = -0.99306148290
tra_pose.pose.orientation.y = 0.0311904773116 
tra_pose.pose.orientation.z = -0.11295288056
tra_pose.pose.orientation.w = 0.009875865653

SPOT_1 = PoseStamped()
SPOT_2 = PoseStamped()
SPOT_3 = PoseStamped()
SPOT_4 = PoseStamped()
SPOT_5 = PoseStamped()


HOME_POSE_T.header.frame_id = 'm1n6s200_link_base'
HOME_POSE_T.pose.position.x = 0.221942096949 
HOME_POSE_T.pose.position.y = -0.253403514624
HOME_POSE_T.pose.position.z = 0.474159002304
HOME_POSE_T.pose.orientation.x = 0.574768424034
HOME_POSE_T.pose.orientation.y = 0.411551713943
HOME_POSE_T.pose.orientation.z = 0.38720113039
HOME_POSE_T.pose.orientation.w = 0.5918967

HOME_POSE_R.header.frame_id = 'm1n6s200_link_base'
HOME_POSE_R.pose.position.x = 0.223211452365 
HOME_POSE_R.pose.position.y = -0.2449244558
HOME_POSE_R.pose.position.z = 0.474389731884
HOME_POSE_R.pose.orientation.x = 0.68142968416
HOME_POSE_R.pose.orientation.y = -0.0933624505997
HOME_POSE_R.pose.orientation.z = -0.00458836555481
HOME_POSE_R.pose.orientation.w = 0.725889801979

PRE_PUSH.header.frame_id = 'm1n6s200_link_base'
PRE_PUSH.pose.position.x = 0.270900969505 
PRE_PUSH.pose.position.y = -0.164824157953
PRE_PUSH.pose.position.z = 0.10402784526
PRE_PUSH.pose.orientation.x = 0.488278537
PRE_PUSH.pose.orientation.y = -0.582844257
PRE_PUSH.pose.orientation.z = 0.484123528004
PRE_PUSH.pose.orientation.w = 0.433013767004

PRE_PUSH_TRANS.header.frame_id = 'm1n6s200_link_base'
PRE_PUSH_TRANS.pose.position.x = 0.1097180867 
PRE_PUSH_TRANS.pose.position.y = -0.294546037912
PRE_PUSH_TRANS.pose.position.z = 0.0702226302
PRE_PUSH_TRANS.pose.orientation.x = -0.991937398911
PRE_PUSH_TRANS.pose.orientation.y = -0.003723668167
PRE_PUSH_TRANS.pose.orientation.z = -0.1242065876
PRE_PUSH_TRANS.pose.orientation.w = 0.024880791083

PRE_PUSH_TRANS_2.header.frame_id = 'm1n6s200_link_base'
PRE_PUSH_TRANS_2.pose.orientation.x = -0.764551699162 
PRE_PUSH_TRANS_2.pose.orientation.y = -0.6444293856 
PRE_PUSH_TRANS_2.pose.orientation.z = -0.012804029509
PRE_PUSH_TRANS_2.pose.orientation.w = -0.00274581089616

PRE_GRASP.header.frame_id = 'm1n6s200_link_base'
PRE_GRASP.pose.position.x = 0.263060510159 
PRE_GRASP.pose.position.y = -0.135446920991
PRE_GRASP.pose.position.z = 0.077095374465
PRE_GRASP.pose.orientation.x = 0.73982489109
PRE_GRASP.pose.orientation.y = -0.0689053833485
PRE_GRASP.pose.orientation.z = 0.00933629274368
PRE_GRASP.pose.orientation.w = 0.669196546078

PRE_GRASP_TRANS1.header.frame_id = 'm1n6s200_link_base'
PRE_GRASP_TRANS1.pose.position.x = -0.318198382854 
PRE_GRASP_TRANS1.pose.position.y = 0.1136691793
PRE_GRASP_TRANS1.pose.position.z = 0.44558733701
PRE_GRASP_TRANS1.pose.orientation.x = 0.455829918385
PRE_GRASP_TRANS1.pose.orientation.y = -0.48101100
PRE_GRASP_TRANS1.pose.orientation.z = -0.57746011018
PRE_GRASP_TRANS1.pose.orientation.w = 0.476851612329

PRE_GRASP_TRANS.header.frame_id = 'm1n6s200_link_base'
PRE_GRASP_TRANS.pose.position.x = -0.29727025985 
PRE_GRASP_TRANS.pose.position.y = -0.131129294634
PRE_GRASP_TRANS.pose.position.z = 0.0411983765662
PRE_GRASP_TRANS.pose.orientation.x = 0.50520116090
PRE_GRASP_TRANS.pose.orientation.y = -0.56314140558
PRE_GRASP_TRANS.pose.orientation.z = 0.444090366364
PRE_GRASP_TRANS.pose.orientation.w = 0.480028480291

WALLET_PRE_GRASP.header.frame_id = 'm1n6s200_link_base'
WALLET_PRE_GRASP.pose.position.x = -0.262734353542 
WALLET_PRE_GRASP.pose.position.y = 0.12135863304
WALLET_PRE_GRASP.pose.position.z = 0.0396186970174
WALLET_PRE_GRASP.pose.orientation.x = 0.706627309322
WALLET_PRE_GRASP.pose.orientation.y = 0.400198042393
WALLET_PRE_GRASP.pose.orientation.z = -0.489282011986
WALLET_PRE_GRASP.pose.orientation.w = 0.317997694016 


SPOT_1.header.frame_id = 'm1n6s200_link_base'
#SPOT_1.header.stamp = rospy.Time.now()
SPOT_1.pose.position.x = -0.297317266464
SPOT_1.pose.position.y = -0.0691142082
SPOT_1.pose.position.z = 0.242255076766 
SPOT_1.pose.orientation.x = 0.994491636753
SPOT_1.pose.orientation.y = -0.0852425396442
SPOT_1.pose.orientation.z = -0.0347545258701
SPOT_1.pose.orientation.w = 0.0501215271652

SPOT_2.header.frame_id = 'm1n6s200_link_base'
#SPOT_2.header.stamp = rospy.Time.now()
SPOT_2.pose.position.x = -0.298411667347
SPOT_2.pose.position.y = 0.00447015557438
SPOT_2.pose.position.z = 0.243017852306
SPOT_2.pose.orientation.x = 0.992346405983
SPOT_2.pose.orientation.y = -0.10713595897
SPOT_2.pose.orientation.z = -0.044642072171
SPOT_2.pose.orientation.w = 0.042161896824

SPOT_3.header.frame_id = 'm1n6s200_link_base'
#SPOT_3.header.stamp = rospy.Time.now()
SPOT_3.pose.position.x = -0.298242390156
SPOT_3.pose.position.y = 0.070592187345
SPOT_3.pose.position.z = 0.24464720487
SPOT_3.pose.orientation.x = 0.991163253784
SPOT_3.pose.orientation.y = -0.116048641
SPOT_3.pose.orientation.z = -0.0563208907843
SPOT_3.pose.orientation.w = 0.0309208687395

SPOT_4.header.frame_id = 'm1n6s200_link_base'
#SPOT_4.header.stamp = rospy.Time.now()
SPOT_4.pose.position.x = -0.294906616211
SPOT_4.pose.position.y = 0.143050059676
SPOT_4.pose.position.z = 0.244550198317
SPOT_4.pose.orientation.x = 0.992886722088
SPOT_4.pose.orientation.y = -0.0996076092124
SPOT_4.pose.orientation.z = -0.0588689595461
SPOT_4.pose.orientation.w = 0.0280831679702

SPOT_5.header.frame_id = 'm1n6s200_link_base'
#SPOT_5.header.stamp = rospy.Time.now()
SPOT_5.pose.position.x = -0.2930341959
SPOT_5.pose.position.y = 0.222065478563
SPOT_5.pose.position.z = 0.244254216552
SPOT_5.pose.orientation.x = 0.991882562637
SPOT_5.pose.orientation.y = -0.107003748417
SPOT_5.pose.orientation.z = -0.0618816465139
SPOT_5.pose.orientation.w = 0.029827259481

SPOT_1j = [337.665435791,
        199.466918945,
        87.7389678955,
        335.93182373,
        73.1590957642,
        39.0681838989,
        0.0]
SPOT_2j = [351.452209473,
        198.198532104,
        86.0294113159,
        336.0,
        72.6136398315,
        50.5909118652,
        0.0]
SPOT_3j = [364.080871582,
        199.191177368,
        88.0147094727,
        335.93182373,
        73.2954559326,
        61.8409118652,
        0.0]
SPOT_4j = [378.419128418,
        203.492645264,
        95.7904434204,
        332.727294922,
        78.5454559326,
        76.5,
        0.0]
SPOT_5j = [391.875,
        211.378677368,
        110.625,
        327.613647461,
        88.5681838989,
        85.0227279663,
        0.0]

