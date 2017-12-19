/* 
 * File:   GraspObject.cpp
 * Author: neha
 * 
 * Created on November 30, 2017, 2:38 PM
 */

#include <vector>
#include <iosfwd>
#include "math.h"
#include "GraspObject.h"

std::string GraspObject::object_property_dir = "g3db_object_labels";
std::string GraspObject::object_pointcloud_dir = "point_clouds";

GraspObject::GraspObject(std::string object_name_, std::string data_dir_name_, bool low_friction, bool load_in_vrep) {
    
    object_name = object_name_;
    data_dir_name = data_dir_name_;
    regression_data_dir_name = data_dir_name_ + "/data_for_regression";
    discretization_step = 0.01;
    if(low_friction)
    {
        min_x_o = min_x_o_low_friction_table;
        default_min_z_o = default_min_z_o_low_friction_table;
        default_initial_object_pose_z = default_initial_object_pose_z_low_friction_table;
        initial_object_x = initial_object_x_low_friction_table;
    }
    else
    {
        min_x_o = min_x_o_high_friction_table;
        default_min_z_o = default_min_z_o_high_friction_table;
        default_initial_object_pose_z = default_initial_object_pose_z_high_friction_table;
        initial_object_x = initial_object_x_high_friction_table;
    }
    min_z_o = default_min_z_o;
    initial_object_pose_z = default_initial_object_pose_z;
    
    //Load object properties file if it exists
    //Initialize python script for loading object
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('python_scripts')");
    PyRun_SimpleString("sys.path.append('scripts')");
    char ** argv;
    //std::cout << "Initialized python 1" << std::endl;
    PySys_SetArgvEx(0, argv, 0); //Required when python script import rospy
    loadObject(load_in_vrep);
}



void GraspObject::loadObject(bool load_in_vrep) {
    
    std::string function_name = "get_object_properties";
    if(load_in_vrep)
    {
        function_name = "add_object_in_scene";
    }
    PyObject* object_properties = callPythonFunction(function_name, object_name, object_property_dir);
    
    std::vector<std::string> property_keys;
    property_keys.push_back("object_min_z");
    property_keys.push_back("object_initial_pose_z");
    
    PyObject* value1;
    for(int i = 0; i < property_keys.size();i++)
    {
        PyObject* key = PyString_FromString(property_keys[i].c_str());
        if(PyDict_Contains(object_properties,key) == 1)
        {
            value1 = PyDict_GetItem(object_properties, key);

            if(i==0)
            {
                min_z_o = PyFloat_AsDouble(value1);
            }
            if(i==1)
            {
                initial_object_pose_z = PyFloat_AsDouble(value1);
            }
        }
        Py_DECREF(key);
    }
    Py_DECREF(object_properties);

}


GraspObject::GraspObject(const GraspObject& orig) {
}

GraspObject::~GraspObject() {
}

void GraspObject::SetObject_name(std::string object_name) {
    this->object_name = object_name;
}

std::string GraspObject::GetObject_name() const {
    return object_name;
}

std::string GraspObject::getRegressionModelDir() {
    return data_dir_name + "/regression_models/" + object_name;
}

std::vector<std::string> GraspObject::getSASOFilenames(bool use_pruned_data, bool discretize_data) {
    std::vector<std::string> ans;
    if(!use_pruned_data && !discretize_data)
    {
        
        
        ans.push_back(data_dir_name + "/SASOData_Cylinder_" + getOldSasoFilename() + "cm_");
        std::cout << "Reading normal files" << std::endl;
                
    }
    else
    {
        std::string data_dir;
        if(discretize_data)
        {
            data_dir = regression_data_dir_name + "/" + object_name;
            std::cout << "Reading discretized files ";
        }
        else
        {
            data_dir = data_dir_name + "/pruned_data_files/" + object_name;
            std::cout << "Reading pruned files ";
        }
        
        PyObject* object_file_list = callPythonFunction("get_pruned_saso_files", object_name, data_dir);
        std::cout << PyList_Size(object_file_list) << std::endl;
        for(int i = 0; i < PyList_Size(object_file_list); i++)
        {
            ans.push_back(PyString_AsString(PyList_GetItem(object_file_list, i)));
        }
        Py_DECREF(object_file_list);
        
    }
    return ans;
}

std::string GraspObject::getOldSasoFilename() {
        
    std::string ans = "";
    std::string prefix  = "Cylinder_";
    std::string values[5] = {"9", "8", "7", "75", "85"};
    for(int i = 0; i < 5; i++)
    {
        if(object_name == prefix + values[i])
        {
            ans = values[i];
            return ans;
        }
        
    }
    //TODO add for coffee and yoghurt cup once their names are finalized
    
    
    return ans;
}

PyObject* GraspObject::callPythonFunction(std::string function_name, std::string arg1, std::string arg2) {
    PyObject *pName;
    pName = PyString_FromString("load_objects_in_vrep");
    PyObject *pModule = PyImport_Import(pName);
    if(pModule == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"load objects in vrep\"\n");
        assert(0==1);
    }
    Py_DECREF(pName);

    
    PyObject *load_function = PyObject_GetAttrString(pModule, function_name.c_str());
    if (!(load_function && PyCallable_Check(load_function)))
    {
        if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"add_object_in_scene\"\n");
    }

    PyObject *pArgs, *pValue;
    pArgs = PyTuple_New(2);
    pValue = PyString_FromString(arg1.c_str());
    /* pValue reference stolen here: */
    PyTuple_SetItem(pArgs, 0, pValue);
    pValue = PyString_FromString(arg2.c_str());
    /* pValue reference stolen here: */
    PyTuple_SetItem(pArgs, 1, pValue);

    PyObject* object_properties = PyObject_CallObject(load_function, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(load_function);
    Py_DECREF(pModule);
    
    return object_properties;
}

std::pair<int, int> GraspObject::getDiscretizationIndex(double x1, double y1) {
    int x1_index = (int)(round(x1/discretization_step));
    int y1_index = (int)(round(y1/discretization_step));
    return std::make_pair(x1_index, y1_index);
}

std::vector<SimulationData> GraspObject::getSimulationData(geometry_msgs::PoseStamped object_pose, geometry_msgs::PoseStamped gripper_pose, int action, bool use_next) {
    std::vector<SimulationData> tempDataVector;
    double x1 = object_pose.pose.position.x - gripper_pose.pose.position.x;
    double y1 = object_pose.pose.position.y - gripper_pose.pose.position.y;
    std::pair<int, int> d_index = getDiscretizationIndex(x1,y1);
    std::vector<int> simulationDataIndices;
    if(use_next)
    {
        simulationDataIndices = discretizedSimulationDataNextState[action][d_index];
    }
    else
    {
        simulationDataIndices = discretizedSimulationDataInitState[action][d_index];
    }
    for(int i = 0; i < simulationDataIndices.size(); i++)
    {
        tempDataVector.push_back(simulationDataCollectionWithObject[action][simulationDataIndices[i]]);
    }
    return tempDataVector;
}







