/* 
 * File:   DeepLearningSolver.h
 * Author: neha
 *
 * Created on August 25, 2016, 1:43 PM
 */

#ifndef DEEPLEARNINGSOLVER_H
#define	DEEPLEARNINGSOLVER_H
#include <despot/core/solver.h>
#include "LearningModel.h"


class DeepLearningSolver : public Solver {
public:
    DeepLearningSolver(const LearningModel* model, Belief* belief = NULL);
    
    ~DeepLearningSolver();

    virtual ValuedAction Search();
    virtual void Update(int action, uint64_t obs);
    
    /*virtual void Update(int action, ObservationClass obs){
        Solver::Update(action, obs.GetHash());
    }*/


private:
    std::string exec(const char* cmd) {
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL)
    		result += buffer;
    }
    pclose(pipe);
    return result;
    }
    
    int deepLearningIdsToModelIds[10] = {6, 4, 7, 3, 1, 9, 2, 8, 5, 0};
};



#endif	/* DEEPLEARNINGSOLVER_H */

