
#include "grasping_box2D.h"
#include <despot/util/floor.h>
#include <math.h>
#include "yaml-cpp/yaml.h"
//#include "grasping.h"
#include <string>

#include "box2d_grasping_world.h"

GraspingBox2D::GraspingBox2D(int start_state_index_) : simWorld() {
 std::cout << "Initializing grasping state Box2D" << std::endl;
    start_state_index = start_state_index_;
    std::cout << "Start state index is " << start_state_index << std::endl;
    if (num_sampled_objects > 5) num_sampled_objects = 5;
    std::vector<int> shuffle_vector;
    for (int i = 1; i < 6; ++i) shuffle_vector.push_back(i);
    //random_shuffle(shuffle_std::vector.begin(), shuffle_vector.end());
    
    //State particles
    for (int i = 0; i < num_sampled_objects; i++) {
        object_id_to_radius.push_back(i+1);
    }
    
    //Belief particles
    for (int i = num_sampled_objects; i <  2*num_sampled_objects; i++) {
        object_id_to_radius.push_back(i+1 - num_sampled_objects);
    }
}

GraspingBox2D::GraspingBox2D(std::string dataFileName, int start_state_index_) : GraspingBox2D(start_state_index_){
    learning_data_file_name = dataFileName;
}

GraspingBox2D::~GraspingBox2D() {

}

bool GraspingBox2D::Step(State& state, double random_num, int action, double& reward, uint64_t& obs) const {

    GraspingStateBox2D& grasping_state = static_cast<GraspingStateBox2D&> (state);
    GraspingObservationBox2D obsClass;
    double circle_radius = object_id_to_radius.at(grasping_state.object_id);
    if (!simWorld.IsValidState(grasping_state)) {
        std::cout << "Step function received an invalid state. This should never happen" << std::endl;
        PrintState(state);
        assert(false);
    }
    reward = step_cost;
    //grasping_state.x_i_change = 0;
    //grasping_state.y_i_change = 0;
    //double initial_x_i = grasping_state.x_i;
    //double initial_y_i = grasping_state.y_i;
    
    bool isTerminal = simWorld.Step(grasping_state, random_num, action, obsClass);
    
    //Update observation class hash
    std::ostringstream obs_string;
    PrintObs(obsClass, obs_string);
    uint64_t hashValue = obsHash(obs_string.str());
    obsClass.SetIntObs(hashValue);
    obs = obsClass.GetHash();
    
    int sensor_obs = obsClass.touch_sensor_reading[0] + obsClass.touch_sensor_reading[1];
    if(sensor_obs != 0)
    {
        //TODO : Adapt this reward function according to box2d
       reward = reward + pow(2,(1 * (1 + (grasping_state.y/15) ) )) - 1;  //Assuming grasping_state.y <=0 when sensor_obs != 0 
    }
    
    return (isTerminal || IsTerminalState(grasping_state));
    
}

double GraspingBox2D::ObsProb(uint64_t obs, const State& state, int action) const {

}



State* GraspingBox2D::CreateStartState(std::string type) const {

        if(start_state_index < 0)
    {
        GraspingStateBox2D* g = new GraspingStateBox2D();
        return g;
    }
    else
    {
        std::cout << "Start_state index is " << start_state_index << std::endl;
        
        std::vector<GraspingStateBox2D*> initial_states =  InitialStartStateParticles();
        std::cout << "Particle size is " <<  initial_states.size()<< std::endl;
        int i = start_state_index % initial_states.size();
        //initial_state = initial_states[i];
        return  initial_states[i];
        
    }
}

std::vector<GraspingStateBox2D*> GraspingBox2D::InitialStartStateParticles(std::string type) const {

}

Belief* GraspingBox2D::InitialBelief(const State* start, std::string type) const {

}

std::vector<State*> GraspingBox2D::InitialBeliefParticles(const State* start, std::string type) const {

}


uint64_t GraspingBox2D::GetObsFromState(State& state, double random_num, int action) const {

}

void GraspingBox2D::PrintObs(const State& state, uint64_t obs, std::ostream& out) const {


}


void GraspingBox2D::PrintObs(uint64_t obs, std::ostream& out) const {


}
void GraspingBox2D::PrintObs(ObservationClass& obs, std::ostream& out) const {

}

double GraspingBox2D::GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const {

}

std::vector<HistoryWithReward*> GraspingBox2D::LearningData() const {

}

void GraspingBox2D::PrintBelief(const Belief& belief, std::ostream& out) const {

}


void GraspingBox2D::PrintState(const State& state, std::ostream& out) const {

}

void GraspingBox2D::PrintAction(int action, std::ostream& out) const {

}



bool GraspingBox2D::IsTerminalState(GraspingStateBox2D state) const {

}

uint64_t GraspingBox2D::GetInitialObs() const {

}

void GraspingBox2D::GenerateAdaboostTestFile(uint64_t obs, History h) const {

}

int GraspingBox2D::GetStartStateIndex() const {
     return start_state_index;
}
