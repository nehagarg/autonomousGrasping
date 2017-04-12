/* 
 * File:   LearningPlanningSolver.cpp
 * Author: neha
 * 
 * Created on September 5, 2016, 3:01 PM
 */

#include "LearningPlanningSolver.h"
#include <despot/core/lower_bound.h>
#include "LearningSolverBase.h"

LearningPlanningSolver::LearningPlanningSolver(const LearningModel* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief)
        : LearningSolverBase(model, belief), despotSolver(model, lb, ub, belief), deepLearningSolver(model),currentSolverLearning(true) {

}

LearningPlanningSolver::~LearningPlanningSolver() {
}

ValuedAction LearningPlanningSolver::Search() {
    ValuedAction ans;
    
    //Hack for running trajectory start
    /*
    if (hist_size == 0) return ValuedAction(1, 1);
    if (hist_size == 1) return ValuedAction(8, 1);
    if (hist_size == 2) return ValuedAction(9, 1);
    return ValuedAction(10,1);
    */        
    //Hack for running trajectory end
    if (currentSolverLearning)
    {
        if(((LearningModel*)model_)->ShallISwitchFromLearningToPlanning(history_))
        {
            currentSolverLearning = false;
        }
    }
    else
    {
        if(((LearningModel*)model_)->ShallISwitchFromPlanningToLearning(history_))
        {
            currentSolverLearning = true;
        }
    }
    if (currentSolverLearning)
    {
        ans =  deepLearningSolver.Search();
        std::cout << "(" << ans.action << "," << ans.value << ")" << std::endl;
        std::cout << "Belief printing from learning planning solver" << std::endl;
        //std::cout << *belief_ << std::endl;
        if (ans.value < 0.0)
        {
            ans =  despotSolver.Search();
        }
    }
    else {
        ans =  despotSolver.Search();
    }
    
    return ans;
}

void LearningPlanningSolver::Update(int action, uint64_t obs) {
    despotSolver.Update(action, obs);
    deepLearningSolver.Update(action, obs);
    history_.Add(action, obs);
    

}

void LearningPlanningSolver::belief(Belief* b) {
    despotSolver.belief(b);
    belief_ = b;
    history_.Truncate(0);
}



