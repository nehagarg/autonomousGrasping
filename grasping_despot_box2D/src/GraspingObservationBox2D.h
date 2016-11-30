/* 
 * File:   GraspingObservation.h
 * Author: neha
 *
 * Created on May 5, 2015, 11:20 AM
 */

#ifndef GRASPINGOBSERVATIONBOX2D_H
#define	GRASPINGOBSERVATIONBOX2D_H

#include "observation.h"

class GraspingObservationBox2D : public ObservationClass {
 public :
        double x;   // x coordinate of gripper w.r.t hand
        double y;   // y coordinate of gripper w.r.t hand
        double finger_joint_state[4];
        double x_i = 0.0; // x coordinate of gripper w.r.t initial position
        double y_i = 0.0; //y coordinate of gripper w.r.t initial position
        //double x_i_change = 0.0; //change in x_i value after taking an action
        //double y_i_change = 0.0; // change in y_i value after taking an action
        int touch_sensor_reading[2];
    
};


#endif	/* GRASPINGOBSERVATIONBOX2D_H */

