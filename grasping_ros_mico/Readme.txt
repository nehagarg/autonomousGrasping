Works with vrep version 3_2_0

to run vrep in headless mode via ssh
xvfb-run --auto-servernum --server-num=1 ./vrep.sh -h ../../WORK_FOLDER/vrep_scenes/micoWithSensorsMutliObjectTrialWithDespotIKVer1.ttt


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
0-100 for generating state therough belonging to 10x10 grid for object locations on table