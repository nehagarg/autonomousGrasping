/* 
 * File:   DeepLearningSolver.cpp
 * Author: neha
 * 
 * Created on August 25, 2016, 11:49 AM
 */

#include "DeepLearningSolver.h"
#include <despot/core/lower_bound.h>

DeepLearningSolver::DeepLearningSolver(const LearningModel* model, Belief* belief) : Solver(model, belief) {
    
    //cout << "Initializing adaboost solver" << endl;
    //Currently id mapping done manually 
    //Ideally should be done by parsing .data file in the model
    /*int a1 = 6; adaboostIdsToModelIds.push_back(a1);
    int a2 = 4; adaboostIdsToModelIds.push_back(a2);
    int a3 = 7; adaboostIdsToModelIds.push_back(a3);
    int a4 = 3; adaboostIdsToModelIds.push_back(a4);
    int a5 = 1; adaboostIdsToModelIds.push_back(a5);
    int a6 = 9; adaboostIdsToModelIds.push_back(a6);
    int a7 = 2; adaboostIdsToModelIds.push_back(a7);
    int a8 = 8; adaboostIdsToModelIds.push_back(a8);
    int a9 = 5; adaboostIdsToModelIds.push_back(a9);
    int a10 = 0; adaboostIdsToModelIds.push_back(a10);
    */
    //cout << "Initialized adaboost solver" << endl;
}


DeepLearningSolver::~DeepLearningSolver() {
    
}
/*template <typename T>
std::string to_string(T const& value) {
    std::stringstream sstr;
    sstr << value;
    return sstr.str();
}*/

ValuedAction DeepLearningSolver::Search() {
    std::cout << "Starting search" << std::endl;
    
   // std::string cmd_string = "cd python_scripts/deepLearning ; python model.py " + to_string((int)((LearningModel*)model_)->GetStartStateIndex()) + "; cd - ;";
    
    std::string cmd_string = ((LearningModel*)model_)->GetPythonExecutionString();
    
    std::cout << "Before calling exec" << std::endl;
    std::string result = exec(cmd_string.c_str());
    std::cout << result << std::endl;
    int action = 0;
    double  value = -100000;
    //parse result and generate action value
    std::istringstream iss(result);
    iss >> action;
    std::cout << "Action is " << action << std::endl;
    int num_actions = model_->NumActions();
    for(int i = 0; i < num_actions; i++)
    {
        double current_value;
        iss >> current_value;
        if (i == action)
        {
            value = current_value;
        }
        
        
    }
    return ValuedAction(action, value);
    //For toy problem
    //return ValuedAction(deepLearningIdsToModelIds[action], value);
    
}

void DeepLearningSolver::Update(int action, uint64_t obs) {
    history_.Add(action, obs);
}



