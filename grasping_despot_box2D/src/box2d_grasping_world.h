/* 
 * File:   box2d_grasping_world.h
 * Author: neha
 *
 * Created on October 25, 2016, 6:33 PM
 */

#ifndef BOX2D_GRASPING_WORLD_H
#define	BOX2D_GRASPING_WORLD_H

#include <Box2D/Box2D.h>
#include "GraspingObservationBox2D.h"
#include "GraspingStateBox2D.h"
#include "box2d_kenv/Box2DBody.h"
#include "box2d_kenv/Box2DLink.h"
#include "box2d_kenv/Box2DJoint.h"
#include "box2d_kenv/Box2DWorld.h"
#include "box2d_kenv/Box2DVisualizer.h"
#include "box2d_kenv/Box2DSensor.h"
#include "box2d_kenv/Box2DSensorMonitor.h"


using box2d_kenv::Box2DBodyPtr;
using box2d_kenv::Box2DLinkPtr;
using box2d_kenv::Box2DJointPtr;
using box2d_kenv::Box2DWorld;
using box2d_kenv::Box2DWorldPtr;
using box2d_kenv::Box2DSensorPtr;
using box2d_kenv::Box2DVisualizer;
using box2d_kenv::Box2DSensorMonitor;

class Box2dGraspingWorld {
public:
    Box2dGraspingWorld();
    virtual ~Box2dGraspingWorld();
    
    bool IsValidState(GraspingStateBox2D state) const;
    bool Step(GraspingStateBox2D& state, double random_num, int action, GraspingObservationBox2D& obs) const;
    
private:
    box2d_kenv::Box2DWorldPtr  world;
    //box2d_kenv::Box2DSensorMonitor const sensor_monitor;
    box2d_kenv::Box2DBodyPtr  ground_body;
    box2d_kenv::Box2DBodyPtr  object_body;
    box2d_kenv::Box2DBodyPtr  hand_body; 
    
};

#endif	/* BOX2D_GRASPING_WORLD_H */

