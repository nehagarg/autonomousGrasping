/* 
 * File:   box2d_grasping_world.cpp
 * Author: neha
 * 
 * Created on October 25, 2016, 6:33 PM
 */

#include "box2d_grasping_world.h"
#include <boost/make_shared.hpp>

Box2dGraspingWorld::Box2dGraspingWorld() {
    
    std::string hand_path = "data/barretthand_twofinger.object.yaml";
    //std::string object_path = "data/ricepilaf.object.yaml";
    std::string object_path = "data/mug.object.yaml";
    double physics_scale = 1000;
    // Setup the physics simulator.
    world = boost::make_shared<Box2DWorld>(physics_scale);
    //sensor_monitor(world);
    ground_body = world->CreateEmptyBody("ground");
    object_body = world->CreateBody("object", object_path);
    hand_body = world->CreateBody("hand", hand_path);

    hand_body->CreateSensors("data/barretthand_fingertip.sensors.yaml");
    

    
}

Box2dGraspingWorld::~Box2dGraspingWorld() {
}

bool Box2dGraspingWorld::Step(GraspingStateBox2D& state, double random_num, int action, GraspingObservationBox2D& obs) const{
    
}

bool Box2dGraspingWorld::IsValidState(GraspingStateBox2D state) const {

}


