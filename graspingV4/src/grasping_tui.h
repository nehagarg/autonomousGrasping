/* 
 * File:   grasping_tui.h
 * Author: neha
 *
 * Created on October 25, 2016, 3:01 PM
 */

#ifndef GRASPING_TUI_H
#define	GRASPING_TUI_H


#include <despot/simple_tui.h>
#include <despot/util/optionparser.h>
#include "grasping_v4.h"
#include "DeepLearningSolver.h"
#include "LearningPlanningSolver.h"
#include "LearningModel.h"

using namespace despot;

class POMDPEvaluatorExt : public POMDPEvaluator {
public:
    POMDPEvaluatorExt(DSPOMDP* model, std::string belief_type,
	Solver* solver, clock_t start_clockt, std::ostream* out,
	double target_finish_time, int num_steps):
        POMDPEvaluator(model, belief_type, solver, start_clockt, out,
	target_finish_time, num_steps)
        {
            
        }
        
    POMDPEvaluatorExt(DSPOMDP* model, std::string belief_type,
	Solver* solver, clock_t start_clockt, std::ostream* out):
        POMDPEvaluator(model, belief_type, solver, start_clockt, out)
        {
            
        }
    bool RunStep(int step, int round) {
        std::cout << *(solver_->belief()) << std::endl;
        return POMDPEvaluator::RunStep(step, round);
    }
    
    bool ExecuteAction(int action, double& reward, OBS_TYPE& obs) {
	double random_num = random_.NextDouble();
	bool terminal = ((LearningModel*)model_)->StepActual(*state_, random_num, action, reward, obs);

	reward_ = reward;
	total_discounted_reward_ += Globals::Discount(step_) * reward;
	total_undiscounted_reward_ += reward;

	return terminal;
    }

};

class TUI: public SimpleTUI {
public:
  TUI() ;
  virtual ~TUI();
  

  virtual void InitializeEvaluator(Evaluator *&simulator,
                                    option::Option *options, DSPOMDP *model,
                                    Solver *solver, int num_runs,
                                    clock_t main_clock_start,
                                    std::string simulator_type, std::string belief_type,
                                    int time_limit, std::string solver_type); 
  
  
  virtual Solver* InitializeSolver(DSPOMDP *model, std::string solver_type,
                                    option::Option *options) ;
 
  
   virtual DSPOMDP* InitializeModel(option::Option* options) = 0;
    /*{
     DSPOMDP* model;
    /*if (options[E_DATA_FILE])
            {
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        
                        model = new GraspingV4(options[E_DATA_FILE].arg, number);
                }
                else
                {
                    model = new GraspingV4(options[E_DATA_FILE].arg, -1);
                }
            }
            else
    
     //{
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        std::cout << "Number is " <<  number;
                        model = new GraspingV4( number);
                }
                else
                {
                    model = new GraspingV4(-1);
                }
     //       }
     
    return model;
  }*/
  
  virtual void InitializeDefaultParameters();

};



  TUI::TUI() {
  }
  
  TUI::~TUI() {}

  void TUI::InitializeEvaluator(Evaluator *&simulator,
                                    option::Option *options, DSPOMDP *model,
                                    Solver *solver, int num_runs,
                                    clock_t main_clock_start,
                                    std::string simulator_type, std::string belief_type,
                                    int time_limit, std::string solver_type) {

      //
  if (time_limit != -1) {
      //std::cout << "If Initializing new evaluator ##############" << std::endl;
    simulator =
        new POMDPEvaluatorExt(model, belief_type, solver, main_clock_start, &std::cout,
                           EvalLog::curr_inst_start_time + time_limit,
                           num_runs * Globals::config.sim_len);
  } else {
      //std::cout << "Else Initializing new evaluator ##############" << std::endl;
    simulator =
        new POMDPEvaluatorExt(model, belief_type, solver, main_clock_start, &std::cout);
  }
}
  
  
  Solver* TUI::InitializeSolver(DSPOMDP *model, std::string solver_type,
                                    option::Option *options) {
      Solver *solver = NULL;
      if(solver_type == "LEARNINGPLANNING")   
      {
          DESPOT *despotSolver = (DESPOT *)InitializeSolver(model, "DESPOT", options);
          solver = new LearningPlanningSolver((LearningModel*)model, despotSolver->lower_bound(), despotSolver->upper_bound(), NULL);
      }
      else if (solver_type == "DEEPLEARNING") {
		solver = new DeepLearningSolver((LearningModel*)model, NULL);
        }
      else {
          solver = SimpleTUI::InitializeSolver(model,solver_type,options);
      }
      return solver;
      
  }
 
  
   
    /*{
     DSPOMDP* model;
    /*if (options[E_DATA_FILE])
            {
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        
                        model = new GraspingV4(options[E_DATA_FILE].arg, number);
                }
                else
                {
                    model = new GraspingV4(options[E_DATA_FILE].arg, -1);
                }
            }
            else
    
     //{
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        std::cout << "Number is " <<  number;
                        model = new GraspingV4( number);
                }
                else
                {
                    model = new GraspingV4(-1);
                }
     //       }
     
    return model;
  }*/
  
  void TUI::InitializeDefaultParameters() {
  }




#endif	/* GRASPING_TUI_H */

