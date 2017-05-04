/* 
 * File:   grasping.h
 * Author: neha
 *
 * Created on September 3, 2014, 10:45 AM
 */

#ifndef GRASPING_V4_H
#define	GRASPING_V4_H

#include <despot/core/pomdp.h>
#include <history_with_reward.h>
#include "LearningModel.h"

class GraspingStateV4 : public State {
    public:
        double x;   // x coordinate of gripper w.r.t hand
        double y;   // y coordinate of gripper w.r.t hand
        double r;   // Width of right side of gripper
        double l;   // Width of left side of gripper
        int object_id ;
        double x_i = 0.0; // x coordinate of gripper w.r.t initial position
        double y_i = 0.0; //y coordinate of gripper w.r.t initial position
        double x_i_change = 0.0; //change in x_i value after taking an action
        double y_i_change = 0.0; // change in y_i value after taking an action
        
    GraspingStateV4() {
    }
 
    GraspingStateV4(double _x, double _y, double _r, double _l, int _object_id) :
        x(_x), y(_y), r (_r), l (_l) , object_id(_object_id), x_i(0.0), y_i(0.0), x_i_change(0.0), y_i_change(0.0){
    }
   
    ~GraspingStateV4() {
    }
    
};


class GraspingV4 : public LearningModel {
public:

    GraspingV4(const GraspingV4& orig);
    //GraspingV4(int start_state_index_);
    GraspingV4(std::string dataFileName, int start_state_index_);
    GraspingV4(int start_state_index_, std::string modelParamFilename = "");
    
    virtual ~GraspingV4();

    int NumActions() const {
        return 10;
    };
    
     /* Deterministic simulative model.*/
    bool Step(State& state, double random_num, int action,
        double& reward, uint64_t& obs) const;
 
    /* Functions related to beliefs and starting states.*/
    double ObsProb(uint64_t obs, const State& state, int action) const;
    Belief* InitialBelief(const State* start, std::string type = "DEFAULT") const;
    State* CreateStartState(std::string type = "DEFAULT") const;
 
    /* Bound-related functions.*/
    double GetMaxReward() const { return reward_max;}
    ValuedAction GetMinRewardAction() const {
        return ValuedAction(A_INCREASE_X, -1);
         
                
    };
    void InitializeScenarioLowerBound(RandomStreams& streams, std::string name);
 
    /* Memory management.*/
    State* Allocate(int state_id, double weight) const {
        //num_active_particles ++;
        GraspingStateV4* state = memory_pool_.Allocate();
        state->state_id = state_id;
        state->weight = weight;
        return state;
    };
    State* Copy(const State* particle) const {
        //num_active_particles ++;
       GraspingStateV4* state = memory_pool_.Allocate();
        *state = *static_cast<const GraspingStateV4*>(particle);
        state->SetAllocated();
        return state;
    };
    void Free(State* particle) const {
        //num_active_particles --;
        memory_pool_.Free(static_cast<GraspingStateV4*>(particle));
    };
    
    int NumActiveParticles() const {
	return memory_pool_.num_allocated();
    }
    
    /**printing functions*/

    void PrintState(const State& state, std::ostream& out = std::cout) const;

    void PrintAction(int action, std::ostream& out = std::cout) const;
    
    void PrintObs(const State& state, uint64_t obs, std::ostream& out = std::cout) const;
    void PrintObs(uint64_t obs, std::ostream& out = std::cout) const;
    
    void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;

    double GetMax_x() const {
        return max_x;
    }

    void SetMax_x(double max_x) {
        this->max_x = max_x;
    }

    double GetMax_y() const {
        return max_y;
    }

    void SetMax_y(double max_y) {
        this->max_y = max_y;
    }

    double GetMin_x()  {
        return min_x;
    }

    void SetMin_x(double min_x) {
        this->min_x = min_x;
    }

    double GetMin_y() const {
        return min_y;
    }

    void SetMin_y(double min_y) {
        this->min_y = min_y;
    }


    enum { //action
        A_INCREASE_X = 0 ,
        A_DECREASE_X = 2 ,
        A_INCREASE_Y = 4 ,
        A_DECREASE_Y = 6,
        A_CLOSE = 8 ,
        A_OPEN = 9
    };
    
    int  num_sampled_objects = 4;
    std::string learning_data_file_name;
    int test_object_id;
    std::vector<int> belief_object_ids;
    int num_belief_particles = 1000;

    mutable MemoryPool<GraspingStateV4> memory_pool_;
    double reward_max = 20;
    double step_cost = -1;
    
    double min_x = -10; //Range for initial belief calculations
    double max_x = +10; // range for initial belief calculations
    double min_y = -15; // range for initial belief calculations
    double max_y = 0; // range for initial belief calculations
    
    double min_x_i = -17; //range for gripper movement
    double max_x_i = 17;  // range for gripper movement
    double min_y_i = -17; // range for gripper movement
    double max_y_i = 17; // range for gripper movement 
    
    int num_gripper_observations = 6; // discretization at 1
    int num_x_i_observations = 35; //discretization at 1
    int num_y_i_observations = 35; // discretization at 1
    int num_x_i_change_observations = 165; // discretization at 0.1  (0 to 16.384/16.4)
    int num_y_i_change_observations = 165; // discretization at 0.1  (0 to 16.384/16.4)
    int start_state_index = -1;
    double max_gripper_width = 5; // actual width twice of this number
    double gripper_length = 10; 
    
    double epsilon = 0.001;
    double min_distance_for_touch = 0.5;
    std::vector<double> object_id_to_radius ;
    //uint64_t initial_obs = -1;
    
    GraspingStateV4* initial_state;
    void GenerateAdaboostTestFile(uint64_t obs, History h) const;
    int GetStartStateIndex() const;
    
    void PrintActionForAdaboost(int action, std::ostream& out = std::cout) const;
    
    
    std::vector<HistoryWithReward*> LearningData() const;
    uint64_t GetInitialObs() const;
    int GetActionIdFromString(std::string dataline ) const;
    double GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const;
    std::vector<State*> InitialBeliefParticles(const State* start, std::string type="DEFAULT") const;
    std::vector<GraspingStateV4*> InitialStartStateParticles(std::string type="DEFAULT") const;
    std::vector< std::pair<double, double> > get_x_range (double gripper_r, double gripper_l, int object_id, double y, bool debug = false ) const;
    std::vector< std::pair<double, double> > get_y_range (double gripper_r, double gripper_l, int object_id, double x ) const;
    double get_x_from_x_range(std::vector< std::pair<double, double> > x_range, double interval, bool debug = false) const ;
    bool IsValidState(double x, double y, double l, double r, int object_id) const;
    bool IsTerminalState(double x, double y, double l, double r, double object_radius) const ;
    std::vector<int> GetObservationBits(uint64_t obs, bool debug = false) const;
    double get_action_range(int action, int action_type) const ;
    double calculate_proprioception_probability(int obs, double value) const ;
    uint64_t GetObsValue(uint64_t sensor_obs, int obs_left_gripper, int obs_right_gripper, int obs_x_i, int obs_y_i, double obs_x_i_change, double obs_y_i_change) const;
    uint64_t GetObsFromState(State& state, double random_num, int action) const; //state is the state reached after performing action
    
    void GetInputSequenceForLearnedmodel(History h, std::ostream& oss) const
    {
        for(int i = 0; i < h.Size(); i++)
            {
                oss << h.Action(i) << ",";
                std::vector<int> obs_binary = GetObservationBits(h.Observation(i), false);
                for (int i = 0; i < 26; i++) {
                    char c = ',';
                    if (i==25){c = '*';}
                    oss << obs_binary[i] << c;
                }
                
            }
        oss << NumActions() ;
        for (int i = 0; i < 26; i++)
        {
            oss << ",-1";
        }
        
        oss << " ";
                
    }
};

#endif	/* GRASPING_V4_H */

