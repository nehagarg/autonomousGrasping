/* 
 * File:   GraspObject.h
 * Author: neha
 *
 * Created on November 30, 2017, 2:38 PM
 */

#ifndef GRASPOBJECT_H
#define	GRASPOBJECT_H

#include "ActionSpecification.h"
#include "simulation_data_reader.h"
#include "ScikitModels.h"
#include "Python.h"
class GraspObject {
public:
    GraspObject(std::string object_name, std::string data_dir_name_, bool low_friction, bool load_in_vrep=false);
    GraspObject(const GraspObject& orig);
    virtual ~GraspObject();
    void SetObject_name(std::string object_name);
    std::string GetObject_name() const;
    std::string getRegressionModelDir();
    std::vector<std::string> getSASOFilenames(bool use_pruned_data);
    void loadObject(bool load_in_vrep = false);
    
    static std::string object_property_dir ;
    static std::string object_pointcloud_dir ;
    
    std::vector<SimulationData> simulationDataCollectionWithObject[A_PICK+1];
    std::vector<int> simulationDataIndexWithObject[A_PICK+1];
    PyObject* dynamicModels[A_PICK+1];
    MultiScikitModels* dynamicModelsC[A_PICK+1];
    
    
    double min_x_o ;
    double max_x_o = 0.5517;  // range for object location
    double min_y_o = 0.0829; // range for object location
    double max_y_o = 0.2295; // range for object location
    double min_z_o; //= 1.1200 ; //for objects on table
    
    double initial_object_x ;
    double initial_object_y = 0.148582;
    double initial_object_pose_z; // = 1.1248; //1.7066; //for amazon shelf
    double default_initial_object_pose_z;
    double default_min_z_o;
    
private:
    
    std::string object_name;
    std::string data_dir_name;
    
    double min_x_o_low_friction_table = 0.4319;
    double min_x_o_high_friction_table = 0.4586;
     //for table with high friction//range for object location
    
    double default_min_z_o_low_friction_table = 1.0950;
    double default_min_z_o_high_friction_table = 1.1200 ; //for objects on high friction table
   
    
    double initial_object_x_low_friction_table = 0.4919;
    double initial_object_x_high_friction_table = 0.498689; 
    
    double default_initial_object_pose_z_high_friction_table = 1.1248; //for object on high friction table
    double default_initial_object_pose_z_low_friction_table = 1.0998;
    
    std::string getOldSasoFilename();
    PyObject* callPythonFunction(std::string function_name, std::string arg1, std::string arg2);
    
    
};

#endif	/* GRASPOBJECT_H */

