/* 
 * File:   LearningPlanningSolver.h
 * Author: neha
 *
 * Created on September 5, 2016, 3:01 PM
 */

#ifndef LEARNINGPLANNINGSOLVER_H
#define	LEARNINGPLANNINGSOLVER_H

#include "LearningSolverBase.h"
#include "LearningModel.h"
#include "DeepLearningSolver.h"
#include <despot/solver/despot.h>



class LearningPlanningSolver : public LearningSolverBase{
public:
    LearningPlanningSolver(const LearningModel* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief = NULL);
            //(const LearningModel* model, Belief* belief, RandomStreams& streams);
    
   // LearningPlanningSolver(const LearningModel* model, DESPOT despotSolver_, DeepLearningSolver deepLearningSolver_, Belief* belief = NULL);

    virtual ~LearningPlanningSolver();
    

    virtual ValuedAction Search();
    virtual void Update(int action, uint64_t obs);
    //virtual void Update(int action, ObservationClass obs);

    virtual void belief(Belief* b);
   

private:
    DESPOT despotSolver;
    DeepLearningSolver deepLearningSolver;
    
};

#endif	/* LEARNINGPLANNINGSOLVER_H */

