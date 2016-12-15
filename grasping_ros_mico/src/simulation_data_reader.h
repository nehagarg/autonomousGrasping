/* 
 * File:   simulation_data_reader.h
 * Author: neha
 *
 * Created on April 30, 2015, 4:41 PM
 */

#ifndef SIMULATION_DATA_READER_H
#define	SIMULATION_DATA_READER_H

#include "geometry_msgs/PoseStamped.h"
#include <fstream>
class SimulationData
{
public:
    geometry_msgs::PoseStamped current_gripper_pose;
    geometry_msgs::PoseStamped current_object_pose;
    double current_finger_joint_state[4];
    geometry_msgs::PoseStamped next_gripper_pose;
    geometry_msgs::PoseStamped next_object_pose;
    double next_finger_joint_state[4];
    geometry_msgs::PoseStamped mico_target_pose;
    double touch_sensor_reading[48];
    
    void PrintSimulationData(std::ostream& out = std::cout);
};


class SimulationDataReader {
public:
    SimulationDataReader();
    SimulationDataReader(const SimulationDataReader& orig);
    virtual ~SimulationDataReader();
    
    void parseSimulationDataLine(std::ifstream& simulationDataFile, SimulationData& simData, int& action, int& reward);
private:

};

#endif	/* SIMULATION_DATA_READER_H */

