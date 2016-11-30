#ifndef GRASPING_BOX2D_H
#define GRASPING_BOX2D_H


#include <despot/core/pomdp.h>
#include "box2d_grasping_world.h"
#include <history_with_reward.h>
#include "LearningModel.h"
#include "GraspingStateBox2D.h"
#include "GraspingObservationBox2D.h"
#include <functional>

class GraspingBox2D :  public LearningModel{
public:
 
    GraspingBox2D(int start_state_index_);
    GraspingBox2D(std::string dataFileName, int start_state_index_);
    
    virtual ~GraspingBox2D();

    int NumActions() const {
        return 7;
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
        GraspingStateBox2D* state = memory_pool_.Allocate();
        state->state_id = state_id;
        state->weight = weight;
        return state;
    };
    State* Copy(const State* particle) const {
        //num_active_particles ++;
       GraspingStateBox2D* state = memory_pool_.Allocate();
        *state = *static_cast<const GraspingStateBox2D*>(particle);
        state->SetAllocated();
        return state;
    };
    void Free(State* particle) const {
        //num_active_particles --;
        memory_pool_.Free(static_cast<GraspingStateBox2D*>(particle));
    };
    
    int NumActiveParticles() const {
	return memory_pool_.num_allocated();
    }
    
    /**printing functions*/

    void PrintState(const State& state, std::ostream& out = std::cout) const;

    void PrintAction(int action, std::ostream& out = std::cout) const;
    
    void PrintObs(const State& state, uint64_t obs, std::ostream& out = std::cout) const;
    void PrintObs(uint64_t obs, std::ostream& out = std::cout) const;
    void PrintObs(ObservationClass& obs, std::ostream& out = std::cout) const;
    
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
        A_DECREASE_X = 1 ,
        A_INCREASE_Y = 2 ,
        A_DECREASE_Y = 3,
        A_CLOSE = 4 ,
        A_OPEN = 5,
        A_PICK = 6
    };
    
    int  num_sampled_objects = 4;
    std::string learning_data_file_name;

    mutable MemoryPool<GraspingStateBox2D> memory_pool_;
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
    
    Box2dGraspingWorld simWorld ;
    
    
    std::hash<std::string> obsHash;
    mutable std::map<uint64_t, GraspingObservationBox2D> obsHashMap;
    GraspingStateBox2D* initial_state;
    void GenerateAdaboostTestFile(uint64_t obs, History h) const;
    int GetStartStateIndex() const;
    
    void PrintActionForAdaboost(int action, std::ostream& out = std::cout) const;
    
    
    std::vector<HistoryWithReward*> LearningData() const;
    uint64_t GetInitialObs() const;
    int GetActionIdFromString(std::string dataline ) const;
    double GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const;
    std::vector<State*> InitialBeliefParticles(const State* start, std::string type="DEFAULT") const;
    std::vector<GraspingStateBox2D*> InitialStartStateParticles(std::string type="DEFAULT") const;
    std::vector< std::pair<double, double> > get_x_range (double gripper_r, double gripper_l, int object_id, double y, bool debug = false ) const;
    std::vector< std::pair<double, double> > get_y_range (double gripper_r, double gripper_l, int object_id, double x ) const;
    double get_x_from_x_range(std::vector< std::pair<double, double> > x_range, double interval, bool debug = false) const ;
    bool IsValidState(GraspingStateBox2D state) const;
    bool IsTerminalState(GraspingStateBox2D state) const ;
    std::vector<int> GetObservationBits(uint64_t obs, bool debug = false) const;
    double get_action_range(int action, int action_type) const ;
    double calculate_proprioception_probability(int obs, double value) const ;
    uint64_t GetObsValue(uint64_t sensor_obs, int obs_left_gripper, int obs_right_gripper, int obs_x_i, int obs_y_i, double obs_x_i_change, double obs_y_i_change) const;
    uint64_t GetObsFromState(State& state, double random_num, int action) const; //state is the state reached after performing action


    
};
 // namespace despot

#endif
