/* 
 * File:   DespotWithLearnedDefaultPolicy.cpp
 * Author: neha
 * 
 * Created on July 13, 2017, 10:12 AM
 */

#include "DespotWithLearnedDefaultPolicy.h"

DespotWithLearnedDefaultPolicy::DespotWithLearnedDefaultPolicy(const LearningModel* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief)
: DESPOT(model, lb, ub, belief),
  deepLearningSolver(model)
{
    std::cout << "Initiallizing despot with learned policy solver ##################" << std::endl;
    deepPolicy = new LearnedPolicy(model_, model_->CreateParticleLowerBound(), belief_, &deepLearningSolver);
    o_helper_ = new DespotStaticFunctionOverrideHelperExt();
}



DespotWithLearnedDefaultPolicy::~DespotWithLearnedDefaultPolicy() {
}

void DespotWithLearnedDefaultPolicy::InitStatistics() {
    //std::cout << "Initiallizing statistics with learned policy solver ##################" << std::endl;
    
    statistics_ = new SearchStatisticsExt();
}

void DespotWithLearnedDefaultPolicy::CoreSearch(std::vector<State*> particles, RandomStreams& streams) {
    //std::cout << "Initiallizing contruct tree with learned policy solver ##################" << std::endl;
    //DESPOT::CoreSearch(particles, streams);
    root_ = ConstructTree(particles, streams, lower_bound_, upper_bound_,
		model_, history_, Globals::config.time_per_move, statistics_, deepPolicy, o_helper_);
}

void DespotWithLearnedDefaultPolicy::Update(int action, OBS_TYPE obs) {
    //std::cout << "Updating belief" << std::endl;
    deepLearningSolver.Update(action, obs);
    //std::cout << "Before Updating despot" << std::endl;
    DESPOT::Update(action,obs);
}

void DespotWithLearnedDefaultPolicy::belief(Belief* b) {
     deepPolicy->belief(b);
     DESPOT::belief(b);     
}

void DespotStaticFunctionOverrideHelperExt::InitMultipleLowerBounds(VNode* vnode, 
        ScenarioLowerBound* lower_bound, RandomStreams& streams, 
        History& history, ScenarioLowerBound* learned_lower_bound,
        SearchStatistics* statistics)
{
    //std::cout << "Initiallizing multiple lower bounds with learned policy solver ##################" << std::endl;
   
    DESPOT::InitLowerBound(vnode,lower_bound, streams,history);
    //double default_policy_lower_bound = vnode->lower_bound();
    double start = clock();
    DESPOT::InitLowerBound(vnode,learned_lower_bound, streams,history);
    double time_used =  double(clock() - start) / CLOCKS_PER_SEC;
    //Update time used by learned policy lower bound calculation so that it can be deducted later
    if (vnode->parent() !=NULL)
    {
        ((SearchStatisticsExt *) statistics)->time_search_learned_policy = 
            ((SearchStatisticsExt *) statistics)->time_search_learned_policy + 
            time_used;
    }
}

double DespotStaticFunctionOverrideHelperExt::GetTimeNotToBeCounted(SearchStatistics* statistics) 
{
    //std::cout << "Calling from learning planning" << std::endl;
    return ((SearchStatisticsExt *) statistics)->time_search_learned_policy;
}

void SearchStatisticsExt::print(std::ostream& os) const
{
    //std::cout << "Printing from ext search statistics      " << std::endl;
    os << "Extra time used by learned policy " << time_search_learned_policy << std::endl;
    SearchStatistics::print(os);
	
}
