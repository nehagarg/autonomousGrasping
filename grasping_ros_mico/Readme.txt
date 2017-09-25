Works with vrep version 3_3_2

to run vrep in headless mode via ssh
xvfb-run --auto-servernum --server-num=1 -s "-screen 0 640x480x24" ./vrep.sh -h ../../WORK_FOLDER/vrep_scenes/micoWithSensorsMutliObjectTrialWithDespotIKVer1.ttt
Remove the port in remoteapiconnections.txt for running multiple vrep instances 
if not using remote api no need to specify new port while runnig vrep, otherwise specify a different remote api port while started a new vrep instance

vrep scenes:
for amazon grasp trial : 
micoWithSensorsAmazonGraspTrialWithDespotIKVer3.ttt

for multi object experiments : 
micoWithSensorsMutliObjectTrialWithDespotIKVer1.ttt ormicoWithSensorsMutliObjectTrialWithDespotIKVer1<Size>Cylinder.ttt (Cylindrical object too heavy. Data collected does not mimic real robot behaviour)
micoWithSensorsMutliObjectTrialWithDespotIKCuboidVer2.ttt(Cuboid object too heavy. Data collected does not mimic real robot behaviour)
micoWithSensorsMutliObjectTrialWithDespotIKYCBObjectsVer3.ttt

micoWithSensorsMutliObjectTrialWithDespotIKVer4.ttt : (Object closer to real arm and gripper behaviour close to real arm behaviour around object)
table friction material : floor material in vrepsc
objecct height : 10cm
object weight : 0.3027 kg

micoWithSensorsMutliObjectTrialWithDespotIKVer5.ttt : (Difference from Ver4 : Gripper closing behaviour closer to real gripper closing behaviour. Gripper palm can detect collisions. Robot position shifted as previous position resulted in instability during pick at far x locations.
To change the scene from Ver4 to Ver 5 : change the force of MicoHand_finger12_motor1 to 4. Change the maximum angles on MicoHand_joint1_finger1/3 to 90, Divide the velocity in Cup script for j1 by 4.0 and Decimate the MicoHand shape by 90% so that it becomes a smple shape from multishape. Set Mico positon to 0.07,0.01





For data collection:
Run vrep 
Then:
1. Generate joint data for various gripper locations using GatherJointData function
2. Generate sas'or data for all actions without object (Not needed for open table as default is fine)
2.1 Use the above data to generate sensor observation for open gripper (For open table can get just one value as gripper values do not change because f enountering shelf )
3. Generate gripper sensor observations with gripper closed without object
4. Generate gripper sensor observations with gripper closed with object
5. Calculate touch threshold for defining touch using values in 2. 3. 4.
6. Generate sas'or data with object for all action till close actions using GatherData() function. It will use the touch threshold calculated in ste 5 for stop untl touch actions.
7.  Generate sas'or data with object for open action using GatherData() function. It is required as we simply cannot use reverse state of close action for open action. Object may fall after opening gripperafter closing gripper.


For despot
-v 3 -t <no of seconds> -n < no of scenarios to sample> -l CAP --belief=<belief type> --number = <number type>

belief type :
SINGLE_PARTICLE : Belief contains only true state
GAUSSIAN : Belief contains 50 particles in gaussian distribution
GAUSSIAN_WITH_STATE_IN : Belief contains 50 particles in gaussian distribution and true state

number type:
-1 for generating state through gaussian distribution
0-49 for generating state therough belonging to 7x7 grid for object locations on table
