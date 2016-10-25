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
        : LearningSolverBase(model, belief), despotSolver(model, lb, ub, belief), deepLearningSolver(model) {

}

LearningPlanningSolver::~LearningPlanningSolver() {
}

ValuedAction LearningPlanningSolver::Search() {
    ValuedAction ans;
    int hist_size = history_.Size();
    if ((hist_size/20) % 2 == 0)
    {
        ans =  deepLearningSolver.Search();
        std::cout << "(" << ans.action << "," << ans.value << ")" << std::endl;
        std::cout << "Belief printing from learning planning solver" << std::endl;
        std::cout << *belief_ << std::endl;
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



