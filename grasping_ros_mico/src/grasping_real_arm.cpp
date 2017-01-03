/* 
 * File:   grasping.cpp
 * Author: neha
 * 
 * Created on September 3, 2014, 10:45 AM
 */

#include "grasping_real_arm.h"
#include <despot/util/floor.h>
#include <despot/core/lower_bound.h>
#include <math.h>
#include "yaml-cpp/yaml.h"
#include "grasping_v4_particle_belief.h"

#include "grasping_ros_mico/Belief.h"
#include "grasping_ros_mico/State.h"
#include "Display/parameters.h"

#include <string>
#include "boost/bind.hpp"  

GraspingRealArm::GraspingRealArm(int start_state_index_, int interfaceType) {
 
     std::cout << "Initializing grasping real arm with interface type" << interfaceType <<  std::endl;
     start_state_index = start_state_index_;  
     if(interfaceType == 0)
     {
        VrepInterface* vrepInterfacePointer = new VrepInterface();
    //
        robotInterface = vrepInterfacePointer;
     }
     
     if(interfaceType == 1)
     {
         std::cout << "Initializing Vrep Data Interface" << std::endl;
        
        VrepDataInterface* interfacePointer = new VrepDataInterface(start_state_index_);
    //
        robotInterface = interfacePointer;
     }
     
     if(interfaceType == 2)
     {
        RealArmInterface* interfacePointer = new RealArmInterface();
    //
        robotInterface = interfacePointer;
     }
     
     
     // Display the belief partilces
    pub_gripper = grasping_display_n.advertise<grasping_ros_mico::State>("gripper_pose", 10);
    pub_belief = grasping_display_n.advertise<grasping_ros_mico::Belief>("object_pose", 10);
     
     //Calling this constructor gives segmentation fault on calling robotinterface functions
     //TODO: investigate why
    //GraspingRealArm(start_state_index_, vrepInterfacePointer);
}

GraspingRealArm::GraspingRealArm(int start_state_index_, VrepInterface* robotInterface_) {
   
    start_state_index = start_state_index_;  
    robotInterface = robotInterface_;
    
}


GraspingRealArm::GraspingRealArm(std::string dataFileName, int start_state_index_) {
    
    start_state_index = start_state_index_;
    /*for (int i = 0; i < num_sampled_objects; i++) {
        object_id_to_radius.push_back(i + 0.5);
    }
    for (int i = num_sampled_objects; i < num_sampled_objects + 5; i++) {
        object_id_to_radius.push_back(i+1 - num_sampled_objects);
    }*/
    learning_data_file_name = dataFileName;
    
    
}

/*vector<HistoryWithReward*> GraspingRealArm::LearningData() const {
    
    vector<HistoryWithReward*> ans;
    
    //Load learning data
    YAML::Node config_full = YAML::LoadFile(learning_data_file_name);
    std::cout << "Yaml file loaded" << std::endl;
    int num_simulations = 0;
    for(int i = 0; i < config_full.size();i++)
    {
        if (num_simulations == 20)
        {
           // break;
        }
        //std::cout << "Config " << i << std::endl;
        //std::cout << config_full[i].Type() << std::endl;
        YAML::Node config = YAML::Load(config_full[i].as<string>());
        //std::cout << config.Type() << std::endl;
        if(config["stepInfo"].size() < 90)
        {
            num_simulations++;
            //std::cout << "stepinfo " << i << std::endl;
            //Get initial obs from initial state
            double g_l = config["roundInfo"]["state"]["g_l"].as<double>();
            double g_r = config["roundInfo"]["state"]["g_r"].as<double>();
            double x_o = config["roundInfo"]["state"]["x_o"].as<double>();
            double y_o = config["roundInfo"]["state"]["y_o"].as<double>();
            int o_id = config["roundInfo"]["state"]["o_r"].as<double>() + num_sampled_objects - 1;
            GraspingStateRealArm* grasping_state = new GraspingStateRealArm(x_o,y_o, g_r,g_l, o_id);
            double random_num = Random::RANDOM.NextDouble();
            uint64_t obs = GetObsFromState(*grasping_state, random_num, 0);
            
            double cummulative_reward = 0;
            HistoryWithReward* h = new HistoryWithReward();
            h->SetInitialObs(obs);
            h->SetInitialState(grasping_state);
            for(int j = 0; j < config["stepInfo"].size(); j++)
            {
                int action = GetActionIdFromString(config["stepInfo"][j]["action"].as<string>());
                
                //Get obs
                uint64_t obs;
                uint64_t sensor_obs = 0;
                for (int k = 0; k < 22; k++)
                {
                    sensor_obs = sensor_obs + (config["stepInfo"][j]["obs"]["sensor_obs"][k].as<int>()*pow(2,k));
                    
                }
                int gripper_l_obs = config["stepInfo"][j]["obs"]["gripper_l_obs"].as<int>();
                int gripper_r_obs = config["stepInfo"][j]["obs"]["gripper_r_obs"].as<int>();
                int terminal_state_obs = config["stepInfo"][j]["obs"]["terminal_state_obs"].as<int>();
                int x_change_obs = config["stepInfo"][j]["obs"]["x_change_obs"].as<int>();
                int y_change_obs = config["stepInfo"][j]["obs"]["y_change_obs"].as<int>();
                int x_w_obs = config["stepInfo"][j]["obs"]["x_w_obs"].as<int>();
                int y_w_obs = config["stepInfo"][j]["obs"]["y_w_obs"].as<int>();
                
                if(terminal_state_obs == 0)
                {
                    obs = GetObsValue(sensor_obs, gripper_l_obs, gripper_r_obs, x_w_obs, y_w_obs, x_change_obs, y_change_obs);
                }
                else
                {
                    obs = pow(2, 22)*num_gripper_observations * num_gripper_observations * num_x_i_observations * num_y_i_observations * num_x_i_change_observations * num_y_i_change_observations;
                }
                
                //Get cummulative reward
                cummulative_reward = cummulative_reward + config["stepInfo"][j]["reward"].as<double>();
                
                h->Add(action, obs, cummulative_reward);
            }
            ans.push_back(h);
        }
        
    }
    for(int i = 0; i < ans.size(); i++)
    {    
        //ans[i]->Print();
    }
    std::cout << ans.size() << " simulations loaded" << std::endl;
    return ans;
    
}
*/
template <typename T>
std::string to_string(T const& value) {
    std::stringstream sstr;
    sstr << value;
    return sstr.str();
}



/*double GraspingRealArm::GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const {
    //std::cout << "Calculating distance between (" << action1 <<  "," << obs1 << ") and (" << action2 <<  "," << obs2 << ")" << std::endl;
    //Distance between actions
    double action_distance = 0;
    double max_action_distance = 30;
    int action1_class = floor(action1/15);
    int action2_class = floor(action2/15);
    if(action1_class == action2_class)
    {
        if(action1_class < 4)
        {
            action_distance = 0.001*(pow(2,action1) - pow(2,action2));
            if(action_distance < 0)
            {
                action_distance = -1 * action_distance;
            }
        }
        else
        {
            if(action1 != action2) //Open gripper close gripper
            {
                action_distance = max_action_distance;
            }
        }
    }
    else
    {
        action_distance = max_action_distance;
    }
    //std::cout << "Action distance:" << action_distance << std::endl; 
    
    //Distance between observations
    vector<int> obs1_bits = GetObservationBits(obs1);
    vector<int> obs2_bits = GetObservationBits(obs2);
    
    double obs_distance = 0;
    for (int i = 0; i < obs1_bits.size(); i++)
    {
        double d = obs1_bits[i] - obs2_bits[i];
        if(d < 0)
        {
            d = -1*d;
        }
        if(i < 22)
        {
            obs_distance = obs_distance + d;
        }
        else if(i < 24) //gripper width
        {
            obs_distance = obs_distance + (d/10.0);
        }
        else if(i < 26) // world coordinates
        {
            obs_distance = obs_distance + (d/34.0);
        }
        else if(i < 28)
        {
            obs_distance = obs_distance + (d/80);
        }
        else if (i < 29)//0 if both are terminal observations
        {
            if(d == 0 )
            {
                if(obs1_bits[i] == 1) // bothe are terminal
                
                {obs_distance = 0;
                }
            }
            else //one is terminal one is not
            {
                obs_distance = 100; //Temporary hack as terminal observation has all other observations 0. /should be remived after getting corrected simulations
            }
        }
    }
   
    
    
    
    
    
    
    return action_distance + (10*obs_distance);
    
}
*/

GraspingRealArm::~GraspingRealArm() {
}






bool GraspingRealArm::StepActual(State& state, double random_num, int action,
        double& reward, ObservationClass& obs) const {
    //std::cout << "." << std::endl;
    GraspingStateRealArm& grasping_state = static_cast<GraspingStateRealArm&> (state);
    //GraspingObservation& grasping_obs = static_cast<GraspingObservation&> (obs);
    
    GraspingObservation grasping_obs;
    
    //ros::Rate loop_rate(10); 
    bool ans = robotInterface->StepActual(grasping_state, random_num, action, reward, grasping_obs);
     
    
    
    //Update observation class hash and store in hashMap
    std::ostringstream obs_string;
    PrintObs(grasping_obs, obs_string);
    uint64_t hashValue = obsHash(obs_string.str());
    obs.SetIntObs(hashValue);
    obsHashMap[hashValue] = grasping_obs;
    
    return ans;
     
    
 
}
/* Deterministic simulative model.*/
bool GraspingRealArm::Step(State& state, double random_num, int action,
        double& reward, ObservationClass& obs) const {
        
    //std::cout << " Starting step \n";
    double step_start_t = get_time_second();
    GraspingStateRealArm& grasping_state = static_cast<GraspingStateRealArm&> (state);
    GraspingObservation grasping_obs;
    
    bool ans = robotInterface->Step(grasping_state, random_num, action, reward, grasping_obs);
    //std::cout << "Reward " << reward << std::endl;
    //Update observation class hash
    std::ostringstream obs_string;
    PrintObs(grasping_obs, obs_string);
    uint64_t hashValue = obsHash(obs_string.str());
    obs.SetIntObs(hashValue);
    //Not storing the hash value in map as the observation returned is not compared using ObsProb function
    //Need to do so in StepActual
    
    double step_end_t = get_time_second();
   // std::cout << "Step took " << step_end_t - step_start_t << std::endl;
    //Decide if terminal state is reached
   
    return ans;
 
}







/* Functions related to beliefs and starting states.*/
double GraspingRealArm::ObsProb(ObservationClass obs, const State& state, int action) const {
    const GraspingStateRealArm& grasping_state = static_cast<const GraspingStateRealArm&> (state);
    std::map<uint64_t,GraspingObservation>::iterator it = obsHashMap.find(obs.GetHash());
    if(it == obsHashMap.end())
    {
        std::cout << "Obs not in hash map. This should not happen" <<  std::endl;
        assert(false);
    }
    GraspingObservation grasping_obs = it->second;
    
    return robotInterface->ObsProb(grasping_obs, grasping_state, action);
}


Belief* GraspingRealArm::InitialBelief(const State* start, std::string type) const
{
    std::cout << "Here" << std::endl;
    return new GraspingParticleBelief(InitialBeliefParticles(start,type), this, NULL, false);
}


std::vector<State*> GraspingRealArm::InitialBeliefParticles(const State* start, std::string type) const
 {
    //std::cout << "In initial belief" << std::endl;
    std::vector<State*> particles;
    int num_particles = 0;
    
    //Gaussian belief for gaussian start state
   /* for(int i = 0; i < 50; i++)
    {
        GraspingStateRealArm* grasping_state = static_cast<GraspingStateRealArm*>(Copy(start));
        robotInterface->CreateStartState(*grasping_state, type);
        particles.push_back(grasping_state);
        num_particles = num_particles + 1;
    }
    */
    
    //Single Particle Belief 
    GraspingStateRealArm* grasping_state = static_cast<GraspingStateRealArm*>(Copy(start));
      particles.push_back(grasping_state);
    num_particles = num_particles + 1;
    
     
     
    /*
     //belief around position obtained from vision
     GraspingStateRealArm* grasping_state = static_cast<GraspingStateRealArm*>(Copy(start));
    //grasping_state->object_pose.pose.position.y = 0.1516;
    particles.push_back(grasping_state);
    num_particles = num_particles + 1;
    double particle_distance = 0.002;
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 50; j++)
        {
            double x_add = particle_distance*(i+1);
            if(i>=5)
            {
                x_add = -1*particle_distance*(i+1-5);
            }
            double y_add = particle_distance*(j+1);
            if(j>=25)
            {
                y_add = -1*particle_distance*(j+1-25);
            }
            
            
            GraspingStateRealArm* grasping_state = static_cast<GraspingStateRealArm*>(Copy(start));
            grasping_state->object_pose.pose.position.y = 0.1516;
            grasping_state->object_pose.pose.position.x = grasping_state->object_pose.pose.position.x + x_add;
            grasping_state->object_pose.pose.position.y = grasping_state->object_pose.pose.position.y + y_add;
            if(robotInterface->IsValidState(*grasping_state))
            {
               particles.push_back(grasping_state);
               num_particles = num_particles + 1; 
            }
            
        }
    }
    */
    /*
    //Belief for data interface
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
           

            GraspingStateRealArm* grasping_state = static_cast<GraspingStateRealArm*>(Copy(start));
            grasping_state->object_pose.pose.position.y = robotInterface->min_y_o + (j*(robotInterface->max_y_o - robotInterface->min_y_o)/9.0);
            grasping_state->object_pose.pose.position.x = robotInterface->min_x_o + (i*(robotInterface->max_x_o - robotInterface->min_x_o)/9.0);
            if(robotInterface->IsValidState(*grasping_state))
            {
               particles.push_back(grasping_state);
               num_particles = num_particles + 1; 
            }
            
        }
    }
    */
    std::cout << "Num particles : " << num_particles << std::endl;
    for(int i = 0; i < num_particles; i++)
    {
        particles[i]->weight = 1.0/num_particles;
    }

    return particles;
}


State* GraspingRealArm::CreateStartState(std::string type) const {
    
    if(initial_state.object_id == -1)
    {
        initial_state.object_id = 0;
        //std::cout << "Creating staet state" << std::endl;
        robotInterface->CreateStartState(initial_state, type);
        
    }
    
    GraspingStateRealArm* grasping_state = static_cast<GraspingStateRealArm*>(Copy(&initial_state));
    

    //cup_display.DrawRviz();
    return  grasping_state;

};


void GraspingRealArm::PrintAction(int action, std::ostream& out) const {
    robotInterface->PrintAction(action, out);
}

void GraspingRealArm::PrintBelief(const Belief& belief, std::ostream& out) const {
    out << "Printing Belief";
}

void GraspingRealArm::PrintObs(ObservationClass& obs, std::ostream& out) const {
  //std::cout << "Before printing observation" << std::endl;
    GraspingObservation grasping_obs;
  std::map<uint64_t,GraspingObservation>::iterator it = obsHashMap.find(obs.GetHash());
    if(it == obsHashMap.end())
    {
        grasping_obs= static_cast<GraspingObservation&> (obs);
    }
    else
    {
        grasping_obs = it->second;
    }
    
  robotInterface->PrintObs(grasping_obs, out);
}


void GraspingRealArm::PrintObs(const State& state, ObservationClass& obs, std::ostream& out) const {

    PrintObs(obs,out);
    PrintState(state, out);
}

void GraspingRealArm::PrintState(const State& state, std::ostream& out) const {
    const GraspingStateRealArm& grasping_state = static_cast<const GraspingStateRealArm&> (state);
    robotInterface->PrintState(grasping_state, out);
    if(out != std::cout)
    {
        out << "\n";
    }
    
}


// Textual display
void GraspingRealArm::DisplayBeliefs(ParticleBelief* belief, 
        std::ostream& ostr) const
{
    
    // sample NUM_PARTICLE_DISPLAY number of particles from the belief set for
    // diaplay purpose
    grasping_ros_mico::Belief msg;
    if(belief->particles().size() > 0)
    {
        msg.numPars = belief->particles().size();
        
        for(int i = 0; i < belief->particles().size(); i++)
        {
            const GraspingStateRealArm& grasping_state = static_cast<const GraspingStateRealArm&> (*(belief->particles()[i]));
             msg.belief.push_back(grasping_state.object_pose.pose.position.x);
             msg.belief.push_back(grasping_state.object_pose.pose.position.y);
             msg.belief.push_back(grasping_state.weight);
        }

    }
    else
        msg.numPars = 0;

    std::cout << "Published belief\n";
    ros::Rate loop_rate(10);
    while(pub_belief.getNumSubscribers() == 0)
    {
        loop_rate.sleep();
    }
    pub_belief.publish(msg);
}
void GraspingRealArm::DisplayState(const State& state, std::ostream& ostr) const
{
    const GraspingStateRealArm& grasping_state = static_cast<const GraspingStateRealArm&> (state);
    if(grasping_state.touch[0])
        std::cout << "left finger in touch" << std::endl;
    if(grasping_state.touch[1])
        std::cout << "right finger in touch" << std::endl;
    if(grasping_state.gripper_status == 0)
        std::cout << "gripper open" << std::endl;
    else if(grasping_state.gripper_status == 1)
        std::cout << "gripper closed but not stable" << std::endl;
    else if(grasping_state.gripper_status == 2)
        std::cout << "gripper closed and stable" << std::endl;
    grasping_ros_mico::State msg;
    msg.gripper_pose = grasping_state.gripper_pose;
    msg.object_pose = grasping_state.object_pose;
    if(grasping_state.gripper_status == 1)
        msg.observation = OBS_NSTABLE;
    else if(grasping_state.gripper_status == 2)
        msg.observation = OBS_STABLE;
    else
        msg.observation = grasping_state.touch[0] * 2 + grasping_state.touch[1];
    std::cout << "Published state\n";
    ros::Rate loop_rate(10);
    while(pub_belief.getNumSubscribers() == 0)
    {
        loop_rate.sleep();
    }
    pub_gripper.publish(msg);
}


class SimpleCloseAndPickPolicy : public Policy {
public:

    SimpleCloseAndPickPolicy(const DSPOMDP* model, ParticleLowerBound* bound)
    : Policy(model, bound) {
    }

    int Action(const std::vector<State*>& particles,
            RandomStreams& streams, History& history) const {
        //std::cout << "Taking action in CAP" << std::endl;
        
        if (history.Size()) {
            if (history.LastAction() == 8) {
                return 10; //10 is Pick 8 is close gripper
            }
        }
        return 8;
    }
};
/*
class GraspingObjectExplorationPolicy : public Policy {
protected:
	const GraspingRealArm* graspingRealArm_;
public:

    GraspingObjectExplorationPolicy(const DSPOMDP* model)
    : Policy(model),
        graspingRealArm_(static_cast<const GraspingRealArm*>(model)){
    }

    int Action(const vector<State*>& particles,
            RandomStreams& streams, History& history) const {
        // 44 is Move up in y with maximum value
        //std::cout << "Taking action" << std::endl;
        GraspingStateRealArm *mostLikelyState = new GraspingStateRealArm(0,0,0,0,0);
        double max_weight = 0;
        for(int i = 0; i < particles.size(); i++)
        {
            if(particles[i]->weight > max_weight)
            {
                max_weight = particles[i]->weight; 
            }
        }
        int object_id[graspingRealArm_->num_sampled_objects];
        for(int  i = 0; i < graspingRealArm_->num_sampled_objects; i++)
        {
            object_id[i] = 0;
        }
        for(int i = 0; i < particles.size(); i++)
        {
            if(particles[i]->weight >= max_weight - 0.0001 )
            {
                GraspingStateRealArm *grasping_state = static_cast<GraspingStateRealArm*> (particles[i]);
                mostLikelyState->x = mostLikelyState->x + (grasping_state->x* particles[i]->weight);
                mostLikelyState->y = mostLikelyState->y + (grasping_state->y* particles[i]->weight);
                mostLikelyState->x_i = mostLikelyState->x_i + (grasping_state->x_i* particles[i]->weight);
                mostLikelyState->y_i = mostLikelyState->y_i + (grasping_state->y_i* particles[i]->weight);
                mostLikelyState->l = mostLikelyState->l + (grasping_state->l* particles[i]->weight);
                mostLikelyState->r = mostLikelyState->r + (grasping_state->r* particles[i]->weight);
                object_id[grasping_state->object_id]++;
            }
            
                
        }
        int max_votes = 0;
        for(int  i = 0; i < graspingRealArm_->num_sampled_objects; i++)
        {
            
            if (object_id[i] > max_votes)
            {
                mostLikelyState->object_id = i;
                max_votes = object_id[i];
            }
        }
        
        if(mostLikelyState->y > -15)
        {
            if(mostLikelyState->y > 0)
            {
                if(mostLikelyState->x < 0 && mostLikelyState->x > graspingRealArm_->min_y_i)
                {
                    return graspingRealArm_->A_DECREASE_X + 14;
                }
                if(mostLikelyState->x > 0 && mostLikelyState->x < graspingRealArm_->max_y_i)
                {
                    return graspingRealArm_->A_INCREASE_X + 14;
                }
                return graspingRealArm_->A_DECREASE_Y + 14;
            }
            else
            {
                if(mostLikelyState->x + mostLikelyState->r < 0 || mostLikelyState->x - mostLikelyState->l > 0)
                {
                    return graspingRealArm_->A_DECREASE_Y + 14;
                }
                else
                {
                    return graspingRealArm_->A_CLOSE;
                }
                    
            }
        }
        else
        {
            if(mostLikelyState->x > -0.001 && mostLikelyState->x < 0.001)
            {
                return graspingRealArm_->A_INCREASE_Y + 14;
            }
            else
            {
                double dist = mostLikelyState->x;
                if(mostLikelyState->x < 0)
                {
                     dist = 0 - mostLikelyState->x;
                }
                int steps = floor(log(dist/0.001)/log(2)) ;
                if(steps > 14) steps = 14;
                if(mostLikelyState->x < 0)
                {
                    return graspingRealArm_->A_INCREASE_X + steps;
                }
                else
                {
                    return graspingRealArm_->A_DECREASE_X + steps;
                }
            }
        }
        
        return 44;
    }
};
*/
ScenarioLowerBound* GraspingRealArm::CreateScenarioLowerBound(std::string name, std::string particle_bound_name ) const {
    if (name == "TRIVIAL" || name == "DEFAULT") {
        //the default lower bound from lower_bound.h
        return new TrivialParticleLowerBound(this);
    } else if (name == "CAP") {
        //set EAST as the default action for each state
        return new SimpleCloseAndPickPolicy(this, new TrivialParticleLowerBound(this));
    }
    //else if (name == "OE") {
        //set EAST as the default action for each state
        //scenario_lower_bound_ = new GraspingObjectExplorationPolicy(this);
    //} 
    else {
        
        std::cerr << "GraspingRealArm:: Unsupported lower bound algorithm: " << name << std::endl;
        exit(0);
        return NULL;
    }
}