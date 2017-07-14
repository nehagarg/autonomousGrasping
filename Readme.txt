To run first create a link to despot home directory
then do 
mkdir build
cd build
cmake ..
make

cd ../grasping_ros_mico
mkdir build
cd build
cmake ..
make

Combined switching strategies
0-x naive threshold based switching with x specifing the threshold
1 svm based switching using 2 svms
2 svm based switching using 1 svm
3-x lower bound based switching with x specifying the learned policy simulation length
4 using learned policy as despot defulat policy

