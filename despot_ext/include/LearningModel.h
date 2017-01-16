/* 
 * File:   LearningModel.h
 * Author: neha
 *
 * Created on September 17, 2015, 4:54 PM
 */

#ifndef LEARNINGMODEL_H
#define	LEARNINGMODEL_H

#include <despot/core/pomdp.h>
#include "history_with_reward.h"



class LearningModel : public DSPOMDP {
public:
    LearningModel();
    virtual ~LearningModel();
    virtual std::vector<HistoryWithReward*> LearningData() const = 0;
    virtual uint64_t GetInitialObs() const = 0;
    virtual double GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const = 0;
    virtual void PrintObs(uint64_t obs, std::ostream& out = std::cout) const = 0;
    virtual void GenerateAdaboostTestFile(uint64_t obs, History h) const;
    virtual int GetStartStateIndex() const;
    virtual bool StepActual(State& state, double random_num, int action, double& reward, uint64_t& obs) const;
    virtual std::string GetPythonExecutionString() const;
};

#endif	/* LEARNINGMODEL_H */

