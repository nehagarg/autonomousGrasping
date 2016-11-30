/* 
 * File:   GraspingStateRealArm.h
 * Author: neha
 *
 * Created on May 5, 2015, 11:18 AM
 */

#ifndef GRASPINGSTATEBOX2D_H
#define	GRASPINGSTATEBOX2D_H

#include <despot/core/pomdp.h>

using namespace despot;

class GraspingStateBox2D : public State {
     public :
        double x;   // x coordinate of gripper w.r.t hand
        double y;   // y coordinate of gripper w.r.t hand
        double finger_joint_state[4];
        double x_i; // x coordinate of gripper w.r.t initial position
        double y_i; //y coordinate of gripper w.r.t initial position
        //double x_i_change = 0.0; //change in x_i value after taking an action
        //double y_i_change = 0.0; // change in y_i value after taking an action
        int object_id;
        
    GraspingStateBox2D() {
        object_id = -1;
    }
 
    GraspingStateBox2D(double _x, double _y, int _object_id) :
        x(_x), y(_y), object_id(_object_id), x_i(0.0), y_i(0.0){
        
    }

    ~GraspingStateBox2D() {
    }
    
};

#endif	/* GRASPINGSTATEBOX2D_H */

