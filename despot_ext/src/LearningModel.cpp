/* 
 * File:   LearningModel1.cpp
 * Author: neha
 * 
 * Created on September 18, 2015, 2:54 PM
 */

#include "LearningModel.h"


LearningModel::LearningModel() {
 

}

LearningModel::~LearningModel() {
}

void LearningModel::GenerateAdaboostTestFile(uint64_t obs, History h) const {

}

int LearningModel::GetStartStateIndex() const {
    return -1;
}

bool LearningModel::StepActual(State& state, double random_num, int action, double& reward, uint64_t& obs) const {
    return Step(state, random_num, action, reward, obs);
}

std::string LearningModel::GetPythonExecutionString() const {
    std::string cmd_string = "cd python_scripts/deepLearning ; python model.py " + to_string((int)(GetStartStateIndex())) + "; cd - ;";
    return cmd_string;
}

std::string LearningModel::GetPythonExecutionString(History h) const {
    return GetPythonExecutionString();
}

double LearningModel::GetUncertaintyValue(Belief* b) const {
    return -1.0;
}

