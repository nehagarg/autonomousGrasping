/* 
 * File:   LearningModel1.cpp
 * Author: neha
 * 
 * Created on September 18, 2015, 2:54 PM
 */

#include "LearningModel.h"
#include "yaml-cpp/yaml.h"


LearningModel::LearningModel() {
 

}

LearningModel::LearningModel(std::string modelParamFileName, std::string problem_name_): problem_name(problem_name_){
    if (modelParamFileName.empty())
    {
        return;
    }
    YAML::Node config = YAML::LoadFile(modelParamFileName);
    if(config["learned_model_name"])
    {
        learned_model_name = config["learned_model_name"].as<std::string>();
    }
    
    if(config["switching_method"])
    {
        automatic_switching_method = config["switching_method"].as<int>();
    }
    
    if(config["svm_model_prefix"])
    {
        svm_model_prefix = config["svm_model_prefix"].as<std::string>();
    }
    
    if(config["switching_threshold"])
    {
        switch_threshold = config["switching_threshold"].as<int>();
    }
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

void LearningModel::GetInputSequenceForLearnedmodel(History h, std::ostream& oss) const {

}

std::string LearningModel::GetPythonExecutionString(History h) const {
    std::ostringstream oss;
        
    oss << "cd python_scripts/deepLearning ; python model.py -p " << problem_name << " -a test -i ";
    GetInputSequenceForLearnedmodel(h, oss);
    oss << "-m " << learned_model_name<< " ; cd - ;" ;
           
    return oss.str();
}

   
std::string LearningModel::GetPythonExecutionStringForJointTraining(History h) const
  {
      std::ostringstream oss;
      oss << "cd python_scripts/deepLearning ; python joint_training_model.py -p " << problem_name << " -a test -i  ";
      GetInputSequenceForLearnedmodel(h, oss);
      oss << "-m " << learned_model_name ;
      oss << " -o " << svm_model_prefix << " ; cd - ;" ;
      return oss.str();
  }
double LearningModel::GetUncertaintyValue(Belief* b) const {
    return -1.0;
}

ValuedAction LearningModel::GetNextActionFromUser(History h) const {
    int next_action;
    std::cout << "Input next action" << std::endl;
    std::cin >> next_action;
    return ValuedAction(next_action, 1);
}

bool LearningModel::ShallISwitchFromLearningToPlanning(History h) const {
    std::cout<< "Asking for switch using method" << automatic_switching_method << std::endl;
        if (automatic_switching_method == 0)
        {
            int hist_size = h.Size();
            return ((hist_size/switch_threshold) % 2 != 0);
        }
        
        std::string command = GetPythonExecutionStringForJointTraining(h);
        std::string result = python_exec(command.c_str());
        std::cout << result << std::endl;
        double seen_scenario_correct;
        double seen_scenario_wrong;
        int seen_scenario;
        std::istringstream iss(result);
        iss >> seen_scenario_correct;
        iss >> seen_scenario_wrong;
        //std::cout << seen_scenario_correct;
        //std::cout << seen_scenario_wrong;
        int seen_scenario_correct_int = (int)seen_scenario_correct;
        int seen_scenario_wrong_int = (int)seen_scenario_wrong;
        
        if (automatic_switching_method == 1)
        {
            if((seen_scenario_correct_int == 1) && (seen_scenario_wrong_int == -1))
            {
                seen_scenario = 1;
            }
            else
            {
                seen_scenario = -1;
            }
        }
        
        if (automatic_switching_method == 2)
        {
            seen_scenario = seen_scenario_correct_int ; //for automatic switching method 2
        }
       
        if (seen_scenario == 1){
            return false;
        }
        else
        {
            return true;
        }
    
}

bool LearningModel::ShallISwitchFromPlanningToLearning(History h) const {
    if(ShallISwitchFromLearningToPlanning(h) == true)
        {
            return false;
        }
        else
        {
            return true;
        }

}

