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
