/* 
 * File:   grasping.cpp
 * Author: neha
 * 
 * Created on September 3, 2014, 10:45 AM
 */

#include "grasping_v4.h"
#include "grasping_v4_particle_belief.h"
#include <despot/util/floor.h>
#include <math.h>
//#include "yaml-cpp/yaml.h"
//#include "grasping.h"
#include <string>

GraspingV4::GraspingV4(int start_state_index_) {
    std::cout << "Initializing grasping state v4" << std::endl;
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

GraspingV4::GraspingV4(std::string dataFileName, int start_state_index_): GraspingV4(start_state_index_) {
    
    /*start_state_index = start_state_index_;
    for (int i = 0; i < num_sampled_objects; i++) {
        object_id_to_radius.push_back(i + 0.5);
    }
    for (int i = num_sampled_objects; i <  2*num_sampled_objects; i++) {
        object_id_to_radius.push_back(i+1 - num_sampled_objects);
    }*/
    //GraspingV4(start_state_index_);
    learning_data_file_name = dataFileName;
    
    
}

int GraspingV4::GetStartStateIndex() const {
    return start_state_index;
}
void GraspingV4::PrintActionForAdaboost(int action, std::ostream& out) const {
        out << "-Action=Actionis";
        if (action == A_CLOSE) {
            out << "CLOSEGRIPPER";
        } else if (action == A_OPEN) {
            out << "OPENGRIPPER";
        } else if (action >= A_DECREASE_Y) {
            out << "DECREASEYby" << (int)get_action_range(action, A_DECREASE_Y);
        } else if (action >= A_INCREASE_Y) {
            out << "INCREASEYby" << (int)get_action_range(action, A_INCREASE_Y);
        } else if (action >= A_DECREASE_X) {
            out << "DECREASEXby" << (int)get_action_range(action, A_DECREASE_X);
        } else if (action >= A_INCREASE_X) {
            out << "INCREASEXby" << (int)get_action_range(action, A_INCREASE_X);
        }
        
         out << ", ";
}

void GraspingV4::GenerateAdaboostTestFile(uint64_t obs, History h) const {
     std::vector<int> obs_bits = GetObservationBits(obs);
     /*
      
     //Grasping verion 1
     ofstream out("grasping.test");
     out << obs_bits[24] << ", " << obs_bits[25] << ", " << obs_bits[22] << ", " << obs_bits[23] << ", " ;
     int num_l_bits = 0;
     int num_r_bits = 0;
     for(int i = 0; i < 11; i++)
     {
         num_l_bits = num_l_bits+ obs_bits[i];
         num_r_bits = num_r_bits + obs_bits[i+11];
     }
     out << num_l_bits << ", " << num_r_bits << ", -Action=ActionisINCREASEXby1.";
     out.close();
     
     //Grasping version2
     ofstream out2("grasping.test");
     out2 << obs_bits[24] << ", " << obs_bits[25] << ", " << obs_bits[22] << ", " << obs_bits[23] << ", " ;
     for(int i = 0; i < 22; i++)
     {
         out2 << obs_bits[i] << ", ";
     }
     out2 << "-Action=ActionisINCREASEXby1.";
     out2.close();
     
     //Grasping v3
     ofstream out("grasping.test");
     out << obs_bits[24] << ", " << obs_bits[25] << ", " << obs_bits[22] << ", " << obs_bits[23] << ", " ;
     int num_l_bits_upper = 0;
     int num_r_bits_upper = 0;
     int num_l_bits_lower = 0;
     int num_r_bits_lower = 0;
     for(int i = 0; i < 6; i++)
     {
         num_l_bits_upper = num_l_bits_upper + obs_bits[i];
         num_r_bits_upper = num_r_bits_upper + obs_bits[i+11];
         num_l_bits_lower = num_l_bits_lower + obs_bits[i+5];
         num_r_bits_lower = num_r_bits_lower + obs_bits[i+16];
     }
     out << num_l_bits_upper << ", " << num_r_bits_upper << ", " << num_l_bits_lower << ", " << num_r_bits_lower << ", -Action=ActionisINCREASEXby1.";
     out.close();
     */
     
     //Grasping version4 
     /*ofstream out("grasping.test");
     if(h.Size() == 0)
     {
         out << "0, 0, ?, ";
     }
     if(h.Size() == 1)
     {
       out << "0, 0, ";   
     }
     if(h.Size() > 1)
     {
        std::vector<int> obs_bits_prev = GetObservationBits(h.Observation(h.Size() - 2).GetHash());
        out << obs_bits_prev[24] << ", " << obs_bits_prev[25] << ", ";  
     }
     if(h.Size() > 0)
     {
         out << "-Action=Actionis";
        int action = h.LastAction();
        if (action == A_CLOSE) {
            out << "CLOSEGRIPPER";
        } else if (action == A_OPEN) {
            out << "OPENGRIPPER";
        } else if (action >= A_DECREASE_Y) {
            out << "DECREASEYby" << (int)get_action_range(action, A_DECREASE_Y);
        } else if (action >= A_INCREASE_Y) {
            out << "INCREASEYby" << (int)get_action_range(action, A_INCREASE_Y);
        } else if (action >= A_DECREASE_X) {
            out << "DECREASEXby" << (int)get_action_range(action, A_DECREASE_X);
        } else if (action >= A_INCREASE_X) {
            out << "INCREASEXby" << (int)get_action_range(action, A_INCREASE_X);
        }
        
         out << ", ";
     }
     
     out << obs_bits[24] << ", " << obs_bits[25] << ", " << obs_bits[22] << ", " << obs_bits[23] << ", " ;
     int num_l_bits = 0;
     int num_r_bits = 0;
     for(int i = 0; i < 11; i++)
     {
         num_l_bits = num_l_bits+ obs_bits[i];
         num_r_bits = num_r_bits + obs_bits[i+11];
     }
     out << num_l_bits << ", " << num_r_bits << ", -Action=ActionisINCREASEXby1.";
     out.close();
     */
     
     //Grasping version 5
     std::ofstream out("grasping.test");
     if(h.Size() == 0)
     {
         out << "0, 0, ?, 0, 0, ?, ";
         std::cout << "0, 0, ?, 0, 0, ?, ";
     }
     if(h.Size() == 1)
     {
       out << "0, 0, ?, 0, 0, ";
       std::cout << "0, 0, ?, 0, 0, ";
       PrintActionForAdaboost(h.LastAction(), out);
       PrintActionForAdaboost(h.LastAction(), std::cout); 
     }
     if(h.Size() == 2)
     {
       out << "0, 0, ";
       std::cout << "0, 0, ";
       PrintActionForAdaboost(h.Action(h.Size() -2), out);
       PrintActionForAdaboost(h.Action(h.Size() -2), std::cout);
       
       std::vector<int> obs_bits_prev = GetObservationBits(h.Observation(h.Size() - 2));
       out << obs_bits_prev[24] << ", " << obs_bits_prev[25] << ", ";
       std::cout << obs_bits_prev[24] << ", " << obs_bits_prev[25] << ", ";
       
       PrintActionForAdaboost(h.LastAction(), out);
       PrintActionForAdaboost(h.LastAction(), std::cout);
       
     }
     
     if(h.Size() > 2)
     {
        std::vector<int> obs_bits_prev2 = GetObservationBits(h.Observation(h.Size() - 3));
        out << obs_bits_prev2[24] << ", " << obs_bits_prev2[25] << ", ";
        std::cout << obs_bits_prev2[24] << ", " << obs_bits_prev2[25] << ", ";
        
        PrintActionForAdaboost(h.Action(h.Size() -2), out);
        PrintActionForAdaboost(h.Action(h.Size() -2), std::cout);
        
        std::vector<int> obs_bits_prev = GetObservationBits(h.Observation(h.Size() - 2));
        out << obs_bits_prev[24] << ", " << obs_bits_prev[25] << ", ";
        std::cout << obs_bits_prev[24] << ", " << obs_bits_prev[25] << ", ";
       
        PrintActionForAdaboost(h.LastAction(), out);
        PrintActionForAdaboost(h.LastAction(), std::cout);
        
     }

     out << obs_bits[24] << ", " << obs_bits[25] << ", " << obs_bits[22] << ", " << obs_bits[23] << ", " ;
     std::cout << obs_bits[24] << ", " << obs_bits[25] << ", " << obs_bits[22] << ", " << obs_bits[23] << ", " ;
     
     int num_l_bits = 0;
     int num_r_bits = 0;
     for(int i = 0; i < 11; i++)
     {
         num_l_bits = num_l_bits+ obs_bits[i];
         num_r_bits = num_r_bits + obs_bits[i+11];
     }
     out << num_l_bits << ", " << num_r_bits << ", -Action=ActionisINCREASEXby1.";
     std::cout << num_l_bits << ", " << num_r_bits << ", -Action=ActionisINCREASEXby1.";
     
     out.close();
     std::cout << "\n";
}

std::vector<HistoryWithReward*> GraspingV4::LearningData() const {
    
    std::vector<HistoryWithReward*> ans;
    //std::cout << "Loading learning data" << std::endl;
    //Load learning data
   /* YAML::Node config_full = YAML::LoadFile(learning_data_file_name);
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
        YAML::Node config = YAML::Load(config_full[i].as<std::string>());
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
            GraspingStateV4* grasping_state = new GraspingStateV4(x_o,y_o, g_r,g_l, o_id);
            double random_num = Random::RANDOM.NextDouble();
            uint64_t obs = GetObsFromState(*grasping_state, random_num, 0);
            //std::cout << "stepinfo " << i << std::endl;
            double cummulative_reward = 0;
            HistoryWithReward* h = new HistoryWithReward();
            h->SetInitialObs(obs);
            h->SetInitialState(grasping_state);
            for(int j = 0; j < config["stepInfo"].size(); j++)
            {
                int action = GetActionIdFromString(config["stepInfo"][j]["action"].as<std::string>());
                
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
    */
    return ans;
    
}

/*template <typename T>
std::string to_string(T const& value) {
    std::stringstream sstr;
    sstr << value;
    return sstr.str();
}
*/
int GraspingV4::GetActionIdFromString(std::string dataline) const {
    int ans = -1;
    std::string action_names[6] = {  "INCREASE X", "DECREASE X", "INCREASE Y", "DECREASE Y", "CLOSE GRIPPER", "OPEN GRIPPER"};
    for (int i = 0; i < 6; i++)
    {
        size_t found = dataline.find(action_names[i]);
        //std::cout << "Before if " << std::endl;
        if(found != std::string::npos)
        {
            //std::cout << "In if " << std::endl;
            if (i < 4)
            {
                for(int j = 0; j < 2; j++)
                {
                    std::string str = to_string(pow(16,j));
                    str.erase ( str.find_last_not_of('0')  , std::string::npos );
                    std::string search_string = action_names[i] + " by " + str;
                    //std::cout << "Search string is : (" << search_std::string << ")" << std::endl;
                    if(dataline.find(search_string) != std::string::npos)
                    {
                        ans = j + (2*i);
                    }
                }
            }
            else
            {
                ans = 4+i;
            }
        }
    }
    if(ans < 0)
    {
        std::cout << "No valid action id ############# found from line " << dataline << std::endl;
        assert(false);
    }
    return ans;
    
}

double GraspingV4::GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const {
    //std::cout << "Calculating distance between (" << action1 <<  "," << obs1 << ") and (" << action2 <<  "," << obs2 << ")" << std::endl;
    //Distance between actions
    double action_distance = 0;
    double max_action_distance = 30;
    int action1_class = floor(action1/2);
    int action2_class = floor(action2/2);
    if(action1_class == action2_class)
    {
        if(action1_class < 4)
        {
            action_distance = 1*(pow(16,action1-(2*action1_class)) - pow(16,action2 - (2*action2_class)));
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
    std::vector<int> obs1_bits = GetObservationBits(obs1);
    std::vector<int> obs2_bits = GetObservationBits(obs2);
    
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


GraspingV4::~GraspingV4() {
}

/* Deterministic simulative model.*/
bool GraspingV4::Step(State& state, double random_num, int action,
        double& reward, uint64_t& obs) const {
    //std::cout << std::endl << "Taking Step " << std::endl;
    //PrintState(state);
    //PrintAction(action);
    //std::cout << "." ;
    GraspingStateV4& grasping_state = static_cast<GraspingStateV4&> (state);
    double circle_radius = object_id_to_radius.at(grasping_state.object_id);
    if (!IsValidState(grasping_state.x, grasping_state.y, grasping_state.l, grasping_state.r, grasping_state.object_id)) {
        std::cout << "Step function received an invalid state. This should never happen" << std::endl;
        PrintState(state);
        assert(false);
    }
    reward = step_cost;
    grasping_state.x_i_change = 0;
    grasping_state.y_i_change = 0;
    if (action == A_CLOSE) {
        //check for terminal state
        if (IsTerminalState(grasping_state.x, grasping_state.y, grasping_state.l, grasping_state.r, circle_radius))
 {
            reward = reward_max;
            obs = pow(2, 22)*num_gripper_observations * num_gripper_observations * num_x_i_observations * num_y_i_observations * num_x_i_change_observations * num_y_i_change_observations;
            obs = GetObsFromState(state, random_num, action) + obs;
            return true;
        } else {
            std::vector< std::pair<double, double> > x_range = get_x_range(0, 0, grasping_state.object_id, grasping_state.y);
            //x_range[0].first = x_range[0].first - max_gripper_width;

            //Left gripper
            int range_index = 0;
            for (int i = 0; i < x_range.size(); i++) {
                if ((grasping_state.x - grasping_state.l) > (x_range[i].second + epsilon)) {
                    range_index++;
                }
            }
            if (grasping_state.x - grasping_state.l + 5 < x_range[range_index].second) {
                grasping_state.l = grasping_state.l - 5;
            } else {
                grasping_state.l = grasping_state.x - x_range[range_index].second;
            }
            if (grasping_state.l < 0) {
                grasping_state.l = 0;
            }

            //Right Gripper
            range_index = 0;
            //x_range[0].first = x_range[0].first + max_gripper_width;
            //x_range[x_range.size() - 1].second = x_range[x_range.size() - 1].second + max_gripper_width;

            for (int i = 0; i < x_range.size(); i++) {
                if ((grasping_state.x + grasping_state.r) > (x_range[i].second + epsilon)) {
                    range_index++;
                }
            }
            if (grasping_state.x + grasping_state.r - 5 > x_range[range_index].first) {
                grasping_state.r = grasping_state.r - 5;
            } else {
                grasping_state.r = x_range[range_index].first - grasping_state.x;
            }
            if (grasping_state.r < 0) {
                grasping_state.r = 0;
            }

        }

    } else if (action == A_OPEN) {
        std::vector< std::pair<double, double> > x_range = get_x_range(0, 0, grasping_state.object_id, grasping_state.y);
        //x_range[0].first = x_range[0].first - max_gripper_width;

        //Left gripper
        int range_index = 0;
        for (int i = 0; i < x_range.size(); i++) {
            if ((grasping_state.x - grasping_state.l) > (x_range[i].second + epsilon)) {
                range_index++;
            }
        }
        if (grasping_state.x - grasping_state.l - 5 > x_range[range_index].first) {
            grasping_state.l = grasping_state.l + 5;
        }
        else {
            grasping_state.l = grasping_state.x - x_range[range_index].first;
        }
        if (grasping_state.l > max_gripper_width) {
            grasping_state.l = max_gripper_width;
        }


        //Right Gripper
        range_index = 0;
        //x_range[0].first = x_range[0].first + max_gripper_width;
        //x_range[x_range.size() - 1].second = x_range[x_range.size() - 1].second + max_gripper_width;

        for (int i = 0; i < x_range.size(); i++) {
            if ((grasping_state.x + grasping_state.r) > (x_range[i].second + epsilon)) {
                range_index++;
            }
        }
        if (grasping_state.x + grasping_state.r + 5 < x_range[range_index].second) {
            grasping_state.r = grasping_state.r + 5;
        } else {
            grasping_state.r = x_range[range_index].second - grasping_state.x;
        }
        if (grasping_state.r > max_gripper_width) {
            grasping_state.r = max_gripper_width;
        }
    } else if (action >= A_DECREASE_Y) {
        double action_value = get_action_range(action, A_DECREASE_Y);
        std::vector< std::pair<double, double> >y_range = get_y_range(grasping_state.r, grasping_state.l, grasping_state.object_id, grasping_state.x);
        int range_index = 0;
        for (int i = 0; i < y_range.size(); i++) {
            if (grasping_state.y > (y_range[i].second + epsilon)) {
                range_index++;
            }
        }
        if (grasping_state.y >= (y_range[range_index].first - epsilon)) //if not position is locked in y
        {
            if (grasping_state.y - action_value > y_range[range_index].first) {
                if(grasping_state.y_i - action_value > min_y_i)
                {
                    grasping_state.y_i_change = action_value;
                }
                else
                {
                    grasping_state.y_i_change = grasping_state.y_i - min_y_i;
                }
                
            } else {
                grasping_state.y_i_change = grasping_state.y - y_range[range_index].first;
            }
            if(grasping_state.y_i_change < 0) {grasping_state.y_i_change = 0;}
            grasping_state.y = grasping_state.y - grasping_state.y_i_change;
            grasping_state.y_i = grasping_state.y_i - grasping_state.y_i_change;
        }
    } else if (action >= A_INCREASE_Y) {
        double action_value = get_action_range(action, A_INCREASE_Y);
        std::vector< std::pair<double, double> >y_range = get_y_range(grasping_state.r, grasping_state.l, grasping_state.object_id, grasping_state.x);
        int range_index = 0;
        for (int i = 0; i < y_range.size(); i++) {
            if (grasping_state.y > (y_range[i].second + epsilon)) {
                range_index++;
            }
        }
        if (grasping_state.y >= (y_range[range_index].first - epsilon)) //if not position is locked in y
        {
            if (grasping_state.y + action_value < y_range[range_index].second) {
                if(grasping_state.y_i + action_value < max_y_i)
                {   
                    grasping_state.y_i_change = action_value;
                }
                else
                {
                    grasping_state.y_i_change = max_y_i - grasping_state.y_i;
                }
                
            } else {
                grasping_state.y_i_change = y_range[range_index].second - grasping_state.y;
            }
            if(grasping_state.y_i_change < 0) {grasping_state.y_i_change = 0;}
             grasping_state.y = grasping_state.y + grasping_state.y_i_change;
             grasping_state.y_i = grasping_state.y_i + grasping_state.y_i_change;
        }
        
    } else if (action >= A_DECREASE_X) {
        double action_value = get_action_range(action, A_DECREASE_X);
        std::vector< std::pair<double, double> > x_range = get_x_range(grasping_state.r, grasping_state.l, grasping_state.object_id, grasping_state.y);
        int range_index = 0;
        for (int i = 0; i < x_range.size(); i++) {
            if (grasping_state.x > (x_range[i].second + epsilon)) {
                range_index++;
            }
        }
        if (grasping_state.x >= (x_range[range_index].first - epsilon)) //if not position is locked in x
        {
            if (grasping_state.x - action_value > x_range[range_index].first) {
                if(grasping_state.x_i - action_value > min_x_i)
                {
                    grasping_state.x_i_change = action_value;
                }
                else
                {
                    grasping_state.x_i_change = grasping_state.x_i - min_x_i;
                }
            } else {
                grasping_state.x_i_change = grasping_state.x - x_range[range_index].first;
            }
            if(grasping_state.x_i_change < 0) {grasping_state.x_i_change = 0;}
            grasping_state.x = grasping_state.x - grasping_state.x_i_change;
            grasping_state.x_i = grasping_state.x_i - grasping_state.x_i_change;
        }


    } else if (action >= A_INCREASE_X) {
        //PrintAction(action);
        double action_value = get_action_range(action, A_INCREASE_X);
        std::vector< std::pair<double, double> > x_range = get_x_range(grasping_state.r, grasping_state.l, grasping_state.object_id, grasping_state.y);

        int range_index = 0;
        for (int i = 0; i < x_range.size(); i++) {
            //std::cout << x_range[i].first << "," << x_range[i].second << " ";
            if (grasping_state.x > (x_range[i].second + epsilon)) {
                range_index++;
            }
        }
        if (grasping_state.x >= (x_range[range_index].first - epsilon)) //if not position is locked in x
        {
            //std::cout << action_value << std::endl;
            if (grasping_state.x + action_value < x_range[range_index].second) {
                if(grasping_state.x_i + action_value < max_x_i)
                {   
                    grasping_state.x_i_change = action_value;
                }
                else
                {
                    grasping_state.x_i_change = max_x_i - grasping_state.x_i;
                }
               
            } else {
                grasping_state.x_i_change = x_range[range_index].second - grasping_state.x;
                
            }
            if(grasping_state.x_i_change < 0) {grasping_state.x_i_change = 0;}
             grasping_state.x = grasping_state.x + grasping_state.x_i_change;
             grasping_state.x_i = grasping_state.x_i + grasping_state.x_i_change;
        }
    }









    //std::cout << "Next State" << std::endl;
    //PrintState(state);


    /*obs = 526336;
    double prob = ObsProb(obs,state,action);
    std::cout<< "Prob is: " << prob << std::endl;
    PrintObs(state, obs);
     */


    obs = GetObsFromState(state, random_num, action);
    uint64_t sensor_obs = obs % ((uint64_t)pow(2,22));
    if(sensor_obs != 0)
    {
       reward = reward + pow(2,(1 * (1 + (grasping_state.y/15) ) )) - 1;  //Assuming grasping_state.y <=0 when sensor_obs != 0 
    }
    //PrintObs(state,obs);
    //std::cout << "Reward is " << reward << std::endl;
    return false;
 
}

uint64_t GraspingV4::GetObsFromState(State& state, double random_num, int action) const {
    GraspingStateV4& grasping_state = static_cast<GraspingStateV4&> (state);
    //std::cout << "Getting Obs" << std::endl;
    //PrintState(grasping_state);
    double circle_radius = object_id_to_radius.at(grasping_state.object_id);
    if (!IsValidState(grasping_state.x, grasping_state.y, grasping_state.l, grasping_state.r, grasping_state.object_id)) {
        std::cout << "Step function received an invalid state. This should never happen" << std::endl;
        PrintState(state);
        assert(false);
    }
    
    uint64_t obs = 0;
    
    
    double prob_sum = 0.0;
    std::vector< std::pair<uint64_t, double> > non_zero_obs;
    std::vector< int > non_zero_obs_bits;
    //Touch sensor obs
    //obs = 0;
    uint64_t sensor_obs = 0;
    //Proprioception observation 
    std::vector<int> obs_left_gripper;
    obs_left_gripper.push_back(floor(grasping_state.l));
    if (floor(grasping_state.l) != ceil(grasping_state.l)) {
        obs_left_gripper.push_back(ceil(grasping_state.l));
    }

    std::vector<int> obs_right_gripper;
    obs_right_gripper.push_back(floor(grasping_state.r));
    if (floor(grasping_state.r) != ceil(grasping_state.r)) {
        obs_right_gripper.push_back(ceil(grasping_state.r));
    }
    std::vector<int> obs_x_i;
    obs_x_i.push_back(floor(grasping_state.x_i));
    if (floor(grasping_state.x_i) != ceil(grasping_state.x_i)) {
        obs_x_i.push_back(ceil(grasping_state.x_i));
    }

    std::vector<int> obs_y_i;
    obs_y_i.push_back(floor(grasping_state.y_i));
    if (floor(grasping_state.y_i) != ceil(grasping_state.y_i)) {
        obs_y_i.push_back(ceil(grasping_state.y_i));
    }
    //std::cout << " x_i_change"
    std::vector<int> obs_x_i_change;
    obs_x_i_change.push_back(floor(grasping_state.x_i_change/0.1));
    if (floor(grasping_state.x_i_change/0.1) != ceil(grasping_state.x_i_change/0.1)) {
        obs_x_i_change.push_back(ceil(grasping_state.x_i_change/0.1));
    }
    
    std::vector<int> obs_y_i_change;
    obs_y_i_change.push_back(floor(grasping_state.y_i_change/0.1));
    if (floor(grasping_state.y_i_change/0.1) != ceil(grasping_state.y_i_change/0.1)) {
        obs_y_i_change.push_back(ceil(grasping_state.y_i_change/0.1));
    }
    

    for (int i = 0; i < obs_left_gripper.size(); i++) {
        for (int j = 0; j < obs_right_gripper.size(); j++) {
            for (int k = 0; k < obs_x_i.size(); k++) {
                for (int l = 0; l < obs_y_i.size(); l++) {
                    for (int m = 0; m < obs_x_i_change.size(); m++) {
                        for (int n = 0; n < obs_y_i_change.size(); n++) {
                           obs = GetObsValue(sensor_obs, obs_left_gripper[i], obs_right_gripper[j], obs_x_i[k], obs_y_i[l], obs_x_i_change[m], obs_y_i_change[n]);
                            //std::cout << "Observation :" << obs << " for " << sensor_obs << " " << obs_left_gripper[i] << " " << obs_right_gripper[j] << " " << obs_x_i[k] << " " << obs_y_i[l] << " " << obs_x_i_change[m] << " " << obs_y_i_change[n] << std::endl;
                            //(grasping_state, obs);
                            double obs_prob = ObsProb(obs, grasping_state, action);
                            prob_sum = prob_sum + obs_prob;
                            non_zero_obs.push_back(std::pair<uint64_t, double> (obs, obs_prob));
                        }
                    }
                }
            }
        }
    }


    //prob_sum = ObsProb(0,grasping_state,action);
    //non_zero_obs.push_back(std::pair<uint64_t, double> (obs, prob_sum));

    //std::vector<int> obs_binary = GetObservationBits(sensor_obs);
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 11; i++) {
            double y_coord = grasping_state.y + i;
            double x_coord = grasping_state.x - grasping_state.l;
            if (j > 0) {
                x_coord = grasping_state.x + grasping_state.r;
            }


            double distance = sqrt((x_coord * x_coord) + (y_coord * y_coord)) - circle_radius;
            if (distance < min_distance_for_touch) {
                non_zero_obs_bits.push_back(i + (j * 11));
            }

        }
    }

    //std::cout << "Non zero obs bits : " << non_zero_obs_bits.size() << " random number " << random_num ;
    for (int i = 0; i < non_zero_obs_bits.size(); i++) {
        //std::cout << "Non zero bit position " << non_zero_obs_bits[i] << " " ;
        uint64_t obs_number = pow(2, non_zero_obs_bits[i]);
        int initial_size = non_zero_obs.size();
        for (int j = 0; j < initial_size; j++) {
            uint64_t new_obs = non_zero_obs[j].first + obs_number;
            double obs_prob = ObsProb(new_obs, grasping_state, action);
            non_zero_obs.push_back(std::pair<uint64_t, double> (new_obs, obs_prob));
            prob_sum = prob_sum + obs_prob;
        }

    }
    
    //std::cout << "non zero obs size : " << non_zero_obs.size() << std::endl;
    double min_interval = 0;
    // std::cout << "Random number : " << random_num << " " ; 
    for (int i = 0; i < non_zero_obs.size(); i++) {

        double prob = non_zero_obs[i].second / prob_sum;
        //std::cout << non_zero_obs[i].first << "," << "Prob actual, normalized : " << non_zero_obs[i].second << "," <<  prob << " ";
        //sensor_obs =  non_zero_obs[i].first % ((uint64_t)pow(2,22));
        //PrintObs(state, non_zero_obs[i].first);
        
        if (random_num < prob + min_interval) {
            obs = non_zero_obs[i].first;
            //sensor_obs = obs % ((uint64_t)pow(2,22));
            //PrintObs(state, obs);
            //std::cout << std::endl;
            if (non_zero_obs.size() > 1) {
                //std::cout << std::endl << "Final observation " ;
                //PrintObs(grasping_state, obs);
            }
            break;
        } else {
            min_interval = min_interval + prob;
        }
    }
    
    /*if (IsTerminalState(grasping_state.x, grasping_state.y, grasping_state.l, grasping_state.r, circle_radius))
    {
        obs = obs + pow(2,28);
    }*/
    return obs;
}

/* Functions related to beliefs and starting states.*/
double GraspingV4::ObsProb(uint64_t obs, const State& state, int action) const {
    const GraspingStateV4& grasping_state = static_cast<const GraspingStateV4&> (state);
    double circle_radius = object_id_to_radius.at(grasping_state.object_id);
    if (!IsValidState(grasping_state.x, grasping_state.y, grasping_state.l, grasping_state.r, grasping_state.object_id)) {
        std::cout << "Invalid state. This should never happen " << std::endl;
        PrintAction(action);
        PrintObs(state, obs);
        assert(false);
        return 0;
    }
    //std::cout << std::endl;
    //PrintObs(state, obs);
    //Get observation bits
    std::vector<int> obs_binary = GetObservationBits(obs);

    if (obs_binary[28] > 0) {
        if (IsTerminalState(grasping_state.x, grasping_state.y, grasping_state.l, grasping_state.r, circle_radius)) {
            std::cout << "Returning 1 because of terminal state" << std::endl;
            return 1.0;
        } else {
            
            std::cout << "Terminal observation without terminal state. This should never happen" << std::endl;
            assert(false);
            return 0.0;
        }
    }
    double prob = 1.0;


    // Calculate for proprioception
    prob = prob * calculate_proprioception_probability(obs_binary[22], grasping_state.l);
    prob = prob * calculate_proprioception_probability(obs_binary[23], grasping_state.r);
    prob = prob * calculate_proprioception_probability(obs_binary[24], grasping_state.x_i);
    prob = prob * calculate_proprioception_probability(obs_binary[25], grasping_state.y_i);
    prob = prob * calculate_proprioception_probability(obs_binary[26], grasping_state.x_i_change/0.1);
    prob = prob * calculate_proprioception_probability(obs_binary[27], grasping_state.y_i_change/0.1);
    
    
    /*if(!prob)
    {
        prob = 1.0/15000; // Hack so that particle set is not empty
        prob = 1;
    }*/
    //std::cout << "Proprioception prob : " << prob << std::endl;

    //Calculate for touch sensor
    for (int j = 0; j < 2; j++) { //j = 0 for left gripper bits
        for (int i = 0; i < 11; i++) {
            //std::cout << "Prob is : " << prob << std::endl;
            double y_coord = grasping_state.y + i;
            double x_coord = grasping_state.x - grasping_state.l;
            if (j > 0) {
                x_coord = grasping_state.x + grasping_state.r;
            }

            if (((x_coord > 0 && j == 0) || (x_coord < 0 && j == 1) || (grasping_state.x + grasping_state.r < 0) || (grasping_state.x - grasping_state.l > 0) ||  (grasping_state.y >= 0)) && (obs_binary.at(i + (j * 11)) == 1))// left gripper is on right side of circle or right gripper is on left side
            {
                //Observation cannot be 1 as there are no sensors on outer side of gripper

                return 0;


            } else {
                double distance = sqrt((x_coord * x_coord) + (y_coord * y_coord)) - circle_radius;
                //std::cout << "D:" << distance << " ";
                if (distance < (0 - epsilon)) {
                    std::cout << "Distance < 0 should never happen for a valid state" << std::endl;
                    PrintObs(state, obs);
                    std::cout << "Distance is " << distance << std::endl;
                    assert(false);
                } else {
                    if (distance < 0) {
                        distance = 0;
                    }
                    if (obs_binary.at(i + (j * 11)) == 1) {
                        /*if (distance < min_distance_for_touch) {
                            prob = prob * (1 / exp(distance));
                        } else {
                            return 0; //prob * (1 / exp(10*distance)); //Hack so that particle filter in not empty
                        }*/
                        prob = prob * (1 / exp(2*distance));
                    } else {
                        if ((x_coord > 0 && j == 0) || (x_coord < 0 && j == 1) || (grasping_state.x + grasping_state.r < 0) || (grasping_state.x - grasping_state.l > 0) || (grasping_state.y >= 0)) {
                            prob = prob * 1.0;
                        } else {
                            prob = prob * (1 - (1 / exp(2*distance)));
                            /*if (distance < min_distance_for_touch) {
                                prob = prob * (1 - (1 / exp(distance))); //Hack so that particle filter is not empty
                            }*/
                        }
                    }
                }

            }
        }
    }

     return prob;
}
Belief* GraspingV4::InitialBelief(const State* start, std::string type) const
{
    return new GraspingParticleBelief(InitialBeliefParticles(start,type), this, NULL, false);
}



std::vector<GraspingStateV4*> GraspingV4::InitialStartStateParticles(std::string type) const
 {

    std::vector<GraspingStateV4*> particles;
    int num_x_samples = 10;
    int num_y_samples = 10;
    //int num_particles = num_x_samples * num_y_samples * 1 * 1 * num_sampled_objects;

    //    return new ParticleBelief(particles, this, NULL , false);
     
    //////Hack end //////////////////////

    
    for (int i = 0; i < num_x_samples; i++) {
        for (int j = 0; j < num_y_samples; j++) {
            for (int k = 4; k < 5; k++) {
                for (int l = 4; l < 5; l++) {
                    for (int m = 0; m < num_sampled_objects; m++) {
                        
                        GraspingStateV4* grasping_state = new GraspingStateV4();
                        
                        grasping_state->r = k + 1;
                        grasping_state->l = l + 1;
                        grasping_state->object_id = m ;
                        grasping_state->y = min_y + (j * (max_y - min_y) / (num_y_samples - 1.0));
                        
                        std::vector< std::pair<double, double> > x_range = get_x_range(grasping_state->r, grasping_state->l, grasping_state->object_id, grasping_state->y);
                       
                        grasping_state->x = get_x_from_x_range(x_range, i / (num_x_samples - 1.0));
                        
                        particles.push_back(grasping_state);
                        if (!IsValidState(grasping_state->x, grasping_state->y, grasping_state->l, grasping_state->r, grasping_state->object_id)) {
                            std::cout << "Invalid belief particle. This should never happen" << std::endl;
                            PrintState(*grasping_state);

                            std::cout << "i:" << i << " j:" << j << " k:" << " l:" << l << std::endl;
                            //std::vector< std::pair<double, double> > x_range_debug = get_x_range(grasping_state->r, grasping_state->l, grasping_state->object_id, grasping_state->y, true);

                            std::cout << "X_range ";
                            for (int n = 0; n < x_range.size(); n++) {
                                std::cout << x_range[n].first << "," << x_range[n].second << " ";
                            }
                            std::cout << std::endl;

                            double xx = get_x_from_x_range(x_range, i / (num_x_samples - 1.0), true);
                            std::cout << "Returned Value is : " << xx << std::endl;
                            assert(false);
                        }
                    }
                }
            }
        }
    }
    
    return particles;
}

std::vector<State*> GraspingV4::InitialBeliefParticles(const State* start, std::string type) const
 {
    
    
    
    
    
    //std::cout << "In initial belief" << std::endl;
    std::vector<State*> particles;
    
    //Hack just one particle which is start state in belief
    //GraspingStateV4* grasping_state = static_cast<GraspingStateV4*>(Copy(start));
    //grasping_state->state_id = -1;
    //grasping_state->weight = 1;
    //particles.push_back(grasping_state);
    //return particles;
    /////// Hack end
    int num_x_samples = 10;
    int num_y_samples = 10;
    int num_object_types = num_sampled_objects;
    if(type.compare("SINGLE_OBJECT") == 0)
    {
        num_object_types = 1;
        num_x_samples = 20;
        num_y_samples = 20;
    }
    int num_particles = num_x_samples * num_y_samples * 1 * 1 * num_object_types;
    double p = 1.0 / (num_particles);
    /////Hack Putting possible state particles in belief
    //p = 1.0 / (num_particles + 1);
    if(type.compare("STATE_IN") == 0)
    {
        std::cout << "Adding state in initial belief \n";
        int num_extra_particles = 0;
        for(int m=0; m < num_sampled_objects; m++)
        {
            GraspingStateV4* grasping_state = static_cast<GraspingStateV4*>(Copy(start));
            grasping_state->state_id = -1;
            grasping_state->object_id = m + num_sampled_objects;
            if(IsValidState(grasping_state->x, grasping_state->y, grasping_state->l, grasping_state->r, grasping_state->object_id))
            {
                num_extra_particles++;
                particles.push_back(grasping_state);
            }
        }
        p = 1.0/(num_particles + num_extra_particles);
        for(int i =0; i<particles.size(); i++)
        {
            particles[i]->weight = p;
        }
    }
    //    return new ParticleBelief(particles, this, NULL , false);
     
    //////Hack end //////////////////////
    //std::cout << "Before for loop" << std::endl;
    
    for (int i = 0; i < num_x_samples; i++) {
        for (int j = 0; j < num_y_samples; j++) {
            for (int k = 4; k < 5; k++) {
                for (int l = 4; l < 5; l++) {
                    for (int m = 0; m < num_object_types; m++) {
                        GraspingStateV4* grasping_state = static_cast<GraspingStateV4*> (Allocate(-1, p));
                        grasping_state->r = k + 1;
                        grasping_state->l = l + 1;
                        grasping_state->object_id = m + num_sampled_objects;
                        grasping_state->y = min_y + (j * (max_y - min_y) / (num_y_samples - 1.0));
                        std::vector< std::pair<double, double> > x_range = get_x_range(grasping_state->r, grasping_state->l, grasping_state->object_id, grasping_state->y);
                        grasping_state->x = get_x_from_x_range(x_range, i / (num_x_samples - 1.0));
                        
                        particles.push_back(grasping_state);
                        if (!IsValidState(grasping_state->x, grasping_state->y, grasping_state->l, grasping_state->r, grasping_state->object_id)) {
                            std::cout << "Invalid belief particle. This should never happen" << std::endl;
                            PrintState(*grasping_state);

                            std::cout << "i:" << i << " j:" << j << " k:" << " l:" << l << std::endl;
                            //std::vector< std::pair<double, double> > x_range_debug = get_x_range(grasping_state->r, grasping_state->l, grasping_state->object_id, grasping_state->y, true);

                            std::cout << "X_range ";
                            for (int n = 0; n < x_range.size(); n++) {
                                std::cout << x_range[n].first << "," << x_range[n].second << " ";
                            }
                            std::cout << std::endl;

                            double xx = get_x_from_x_range(x_range, i / (num_x_samples - 1.0), true);
                            std::cout << "Returned Value is : " << xx << std::endl;
                            assert(false);
                        }
                    }
                }
            }
        }
    }
    //std::cout<< "Returning particles" << std::endl;
    return particles;
}
uint64_t GraspingV4::GetInitialObs() const {
    uint64_t initial_obs;
        State* g = CreateStartState();
        initial_obs = GetObsFromState(*g, Random::RANDOM.NextDouble(), 0);
    
    return initial_obs;
}

State* GraspingV4::CreateStartState(std::string type) const {
    //return new GraspingStateV4(min_x, min_y, 5, 5, Random::RANDOM.NextInt(num_sampled_objects));
    if(start_state_index < 0)
    {
        GraspingStateV4* g = new GraspingStateV4(4, -11, 5, 5, 0);
        return g;
    }
    else
    {
        std::cout << "Start_state index is " << start_state_index << std::endl;
        
        std::vector<GraspingStateV4*> initial_states =  InitialStartStateParticles();
        std::cout << "Particle size is " <<  initial_states.size()<< std::endl;
        int i = start_state_index % initial_states.size();
        //initial_state = initial_states[i];
        return  initial_states[i];
        
    }
    //return new GraspingStateV4(-5.692, min_y, 0, 0, 0);
};

bool GraspingV4::IsTerminalState(double x, double y, double l, double r, double object_radius) const {
    double x_ = (x - l + x + r)/2;
    double l_ = l + x_ - x;
    double r_ = r - x_ + x;
    if ((x_ > (0 - epsilon)) &&
            (x_ < (0 + epsilon)) &&
            ((y + gripper_length - object_radius)> (0 - epsilon)) &&
            (y <= (-1*object_radius + epsilon)) &&
            ((r_ - object_radius) < (0 + epsilon)) &&
            ((r_ - object_radius) > (0 - epsilon)) &&
            ((l_ - object_radius) < (0 + epsilon)) &&
            ((l_ - object_radius) > (0 - epsilon))) {
        return true;

    }
    return false;
}

/*
 * Get the range of y values possible for given x coordinate of gripper
 * 
 */
std::vector< std::pair<double, double> > GraspingV4::get_y_range(double gripper_r, double gripper_l, int o_id, double x) const {
    std::vector< std::pair<double, double> > ans;
    double min_allowed = -INFINITY;
    double max_allowed = INFINITY;

    //get minimum value of |x| as y value x will lie maximum in circle
    double min_mod_x = x - gripper_l; //In case x - gripper_l > 0
    if (min_mod_x < 0) {
        if (x + gripper_r >= 0) { //x_coordinate passes through center of circle
            min_mod_x = 0;
        } else {
            min_mod_x = -1 * (x + gripper_r);
        }
    }
    double circle_radius = object_id_to_radius.at(o_id);
    if (circle_radius <= min_mod_x) {
        ans.push_back(std::pair<double, double>(min_allowed, max_allowed));
    } else // get range of forbidden y
    {
        //for palm
        double y_value = sqrt((circle_radius * circle_radius) - (min_mod_x * min_mod_x));
        double y_value_left = -1;
        //for left finger 
        if (((circle_radius - epsilon) * (circle_radius-epsilon)) >= ((x - gripper_l)*(x - gripper_l))) {
            y_value_left = sqrt((circle_radius * circle_radius) - ((x - gripper_l)*(x - gripper_l)));
        }
        //for right finger
        double y_value_right = -1;
        if (((circle_radius - epsilon) * (circle_radius-epsilon)) >= ((x + gripper_r)*(x + gripper_r))) {
            y_value_right = sqrt((circle_radius * circle_radius) - ((x + gripper_r)*(x + gripper_r)));
        }

        if ((y_value_left < 0 ) && (y_value_right < 0 )) {
            ans.push_back(std::pair<double, double>(min_allowed, -1 * y_value));
        } else {
            double bigger_y_value = y_value_left;
            if (y_value_right > bigger_y_value) {
                bigger_y_value = y_value_right;
            }
            if (((-1 * bigger_y_value) - gripper_length) < (-1 * y_value)) {
                ans.push_back(std::pair<double, double>(min_allowed, ((-1 * bigger_y_value) - gripper_length)));
            } else {
                ans.push_back(std::pair<double, double>(min_allowed, -1 * y_value));
            }
        }
        ans.push_back(std::pair<double, double>(y_value, max_allowed));
    }

    return ans;
}

/*
 * Get the range of x values possible for given y coordinate of gripper
 * 
 */
std::vector< std::pair<double, double> > GraspingV4::get_x_range(double gripper_r, double gripper_l, int o_id, double y, bool debug) const {
    std::vector< std::pair<double, double> > ans;
    double min_allowed = -INFINITY;
    double max_allowed = INFINITY;
    //get minimum value of |y| as x value at that y will lie maximum in circle
    double min_mod_y = y; // in case y > 0
    if (y < 0) {
        if ((y + gripper_length) >= 0) { // gripper passes through center of circle
            min_mod_y = 0;
        } else // get minimum value by adding the gripper length
        {
            min_mod_y = -1 * (y + gripper_length);
        }
    }
    if (debug) {
        std::cout << "Min_mod_y :" << min_mod_y << std::endl;
           std::cout << "object is to radius size" << object_id_to_radius.size() << std::endl;
    }
 
    double circle_radius = object_id_to_radius.at(o_id);
    if (circle_radius <= min_mod_y) // any value of x is permissible
    {
        ans.push_back(std::pair<double, double>(min_allowed, max_allowed));
    } else // get range of forbidden x 
    {
        //For fingers
        double x_value = sqrt((circle_radius * circle_radius) - (min_mod_y * min_mod_y));
        if (debug) {
            std::cout << "X_Value :" << x_value << std::endl;
        }
        //range forbidden becasue of collision with left finger
        double forbidden_min_x_l = (-1 * x_value) + gripper_l;
        double forbidden_max_x_l = x_value + gripper_l;

        //range forbidden because of collision with right finger
        double forbidden_min_x_r = (-1 * x_value) - gripper_r;
        double forbidden_max_x_r = x_value - gripper_r;

        //For palm
        double x_value_palm = -1;
        if ((circle_radius * circle_radius) > ((y)*(y))) {
            x_value_palm = sqrt((circle_radius * circle_radius) - (y * y));
        }
        if (debug) {
            std::cout << "X_value_palm : " << x_value_palm << std::endl;
            std::cout << " forbidden_min_x_l" << forbidden_min_x_l << std::endl;
            std::cout << " forbidden_max_x_r" << forbidden_max_x_r << std::endl;
        }
        //push non forbidden range into ans
        ans.push_back(std::pair<double, double>(min_allowed, forbidden_min_x_r));
        if (forbidden_min_x_l >= forbidden_max_x_r) {
            if (x_value_palm < 0) {
                ans.push_back(std::pair<double, double>(forbidden_max_x_r, forbidden_min_x_l));
            }
        }
        ans.push_back(std::pair<double, double>(forbidden_max_x_l, max_allowed));

    }

    return ans;
}

/*
 Divide range of x into intervals specified by interval_multiplier
 * Used to generate initial belief
 */
double GraspingV4::get_x_from_x_range(std::vector< std::pair<double, double> > x_range, double interval_multiplier, bool debug) const {
    double allowed_range_size = 0;

    x_range[0].first = min_x;
    x_range[x_range.size() - 1].second = max_x;

    for (int i = 0; i < x_range.size(); i++) {
        allowed_range_size = allowed_range_size + x_range[i].second - x_range[i].first;
    }

    double x = min_x + (interval_multiplier * allowed_range_size);
    if (debug) {
        std::cout << "Before Iteration Value of x is : " << x << std::endl;
    }
    for (int i = 0; i < x_range.size(); i++) {
        if (debug) {
            std::cout << "Iteration: " << i << " x=" << x << std::endl;
        }
        if (x <= (x_range[i].second + epsilon)) {
            break;
        } else {
            x = x + (x_range[i + 1].first - x_range[i].second);
        }
    }
    if (debug) {
        std::cout << "After Iteration Value of x is : " << x << std::endl;
    }
    return x;


}

//Check is gripper is not colliding with the object

bool GraspingV4::IsValidState(double x, double y, double l, double r, int object_id) const {
    double tolerance = epsilon;
    std::vector< std::pair<double, double> > x_range = get_x_range(r, l, object_id, y);
    for (int i = 0; i < x_range.size(); i++) {

        if (x >= (x_range[i].first - tolerance)) {
            if (x <= (x_range[i].second + tolerance)) {
                return true;
            }
        }
    }
    //To be checked for states which are achieved by moving y
    std::vector< std::pair<double, double> > y_range = get_y_range(r, l, object_id, x);
    for (int i = 0; i < y_range.size(); i++) {

        if (y >= (y_range[i].first - tolerance)) {
            if (y <= (y_range[i].second + tolerance)) {
                return true;
            }
        }
    }

    //debug informaion in case of invalid state
    std::cout << "x = " << x << " y = " << y << "tolerance= " << tolerance << std::endl;
    std::cout << "X_range ";
    for (int n = 0; n < x_range.size(); n++) {
        std::cout << x_range[n].first << "," << x_range[n].second << " ";
    }
    std::cout << std::endl;
    //debug informaion in case of invalid state
    std::cout << "Y_range ";
    for (int n = 0; n < y_range.size(); n++) {
        std::cout << y_range[n].first << "," << y_range[n].second << " ";
    }
    std::cout << std::endl;
    return false;
}

std::vector<int> GraspingV4::GetObservationBits(uint64_t obs, bool debug) const {

    std::vector<int> obs_binary;
    //Touch sensor bits
    for (int i = 0; i < 22; i++) {
        if (obs & 1) {
            obs_binary.push_back(1);
        } else {
            obs_binary.push_back(0);
        }
        obs >>= 1;
    }

    if (debug) {
        std::cout << "Proprioception observation :" << obs << std::endl;
    }

    //Discretized Proprioception observation
    int l_obs = obs % num_gripper_observations;
    obs_binary.push_back(l_obs);
    obs = (obs - l_obs) / num_gripper_observations;

    int r_obs = obs % num_gripper_observations;
    obs_binary.push_back(r_obs);
    obs = (obs - r_obs) / num_gripper_observations;

    int x_obs = obs % num_x_i_observations;
    obs_binary.push_back(x_obs + min_x_i);
    obs = (obs - x_obs) / num_x_i_observations;

    int y_obs = obs % num_y_i_observations;
    obs_binary.push_back(y_obs + min_y_i);
    obs = (obs - y_obs) / num_y_i_observations;

    int x_change_obs = obs % num_x_i_change_observations;
    obs_binary.push_back(x_change_obs );
    obs = (obs - x_change_obs) / num_x_i_change_observations;

    int y_change_obs = obs % num_y_i_change_observations;
    obs_binary.push_back(y_change_obs );
    obs = (obs - y_change_obs) / num_y_i_change_observations;
    obs_binary.push_back(obs);

    return obs_binary;
}

uint64_t GraspingV4::GetObsValue(uint64_t sensor_obs, int obs_left_gripper, int obs_right_gripper, int obs_x_i, int obs_y_i, double obs_x_i_change, double obs_y_i_change) const {
    obs_x_i = obs_x_i - min_x_i;
    obs_y_i = obs_y_i - min_y_i;
    int obs_x_i_change_ = obs_x_i_change ;
    int obs_y_i_change_ = obs_y_i_change ;
    //std::cout << "While getting obs value " << obs_x_i_change_ << "," << obs_y_i_change_ << " " << obs_x_i_change << "," << obs_y_i_change << std::endl; 
    uint64_t ans = obs_y_i_change_*num_x_i_change_observations;
    ans = (ans + obs_x_i_change_) * num_y_i_observations;
    ans = (ans + obs_y_i) * num_x_i_observations;
    ans = (ans + obs_x_i) * num_gripper_observations;
    ans = (ans + obs_right_gripper) * num_gripper_observations;
    ans = (ans + obs_left_gripper)*(pow(2, 22));
    ans = ans + sensor_obs;
    return ans;
}

void GraspingV4::PrintAction(int action, std::ostream& out) const {
    out << "Action is ";
    if (action == A_CLOSE) {
        out << "CLOSE GRIPPER";
    } else if (action == A_OPEN) {
        out << "OPEN GRIPPER";
    } else if (action >= A_DECREASE_Y) {
        out << "DECREASE Y by " << get_action_range(action, A_DECREASE_Y);
    } else if (action >= A_INCREASE_Y) {
        out << "INCREASE Y by " << get_action_range(action, A_INCREASE_Y);
    } else if (action >= A_DECREASE_X) {
        out << "DECREASE X by " << get_action_range(action, A_DECREASE_X);
    } else if (action >= A_INCREASE_X) {
        out << "INCREASE X by " << get_action_range(action, A_INCREASE_X);
    }


    out << std::endl;
}

void GraspingV4::PrintBelief(const Belief& belief, std::ostream& out) const {
    out << "Printing Belief";
}

void GraspingV4::PrintObs(uint64_t obs, std::ostream& out) const {
 out << "Obs is :" << obs << " ";
    //Get observation bits
    std::vector<int> obs_binary = GetObservationBits(obs, true);

    for (int i = 0; i < obs_binary.size(); i++) {
        if (i >= 22) {
            std::cout << " ";
        };
        std::cout << obs_binary[i];
    }
    std::cout << " for State " << std::endl;
}

void GraspingV4::PrintObs(const State& state, uint64_t obs, std::ostream& out) const {

    PrintObs(obs,out);
    PrintState(state, out);
}

void GraspingV4::PrintState(const State& state, std::ostream& out) const {
    const GraspingStateV4& grasping_state = static_cast<const GraspingStateV4&> (state);
    out << "Gripper at: " << grasping_state.x << "," << grasping_state.y << " w.r.t object";
    out << " Opening left: " << grasping_state.l << " Opening Right:" << grasping_state.r;
    out << " Circle radius:" << object_id_to_radius[grasping_state.object_id] << " for object id " << grasping_state.object_id ;
    out << " World coordinates:" << grasping_state.x_i << "," << grasping_state.y_i;
    out << " Change:"<< grasping_state.x_i_change << "," << grasping_state.y_i_change<< std::endl;
}

double GraspingV4::get_action_range(int action, int action_type) const {
    if ((action - action_type) >= (A_DECREASE_X - A_INCREASE_X)) {
        std::cout << "Action " << action << "out of range of action type " << action_type << std::endl;
        assert(false);
    }
    return  pow(16, action - action_type);
}

double GraspingV4::calculate_proprioception_probability(int obs, double value) const {
    double prob = 1.0;
    if ((obs == floor(value))) {
        if (obs == ceil(value)) {
            prob = 1.0;
        } else {
            prob = 0.5 + 0.5 * (ceil(value) - value);
        }
    } else if (obs == ceil(value)) {
        prob = 0.5 + 0.5 * (value - floor(value));
    } else {
       //Assign exponentially decreasing probbilities
        if(obs < floor(value))
        {
            double floor_prob = calculate_proprioception_probability(floor(value), value);
            prob = pow(0.5,floor(value)-obs)*floor_prob;
        }
        if(obs > ceil(value))
        {
            double ceil_prob = calculate_proprioception_probability(ceil(value), value);
            prob = pow(0.5,obs - ceil(value))*ceil_prob;
        }
    }

    return prob;
}
/*
class SimpleGraspingUpPolicy : public Policy {
public:

    SimpleGraspingUpPolicy(const DSPOMDP* model)
    : Policy(model) {
    }

    int Action(const std::vector<State*>& particles,
            RandomStreams& streams, History& history) const {
        // 44 is Move up in y with maximum value
        //std::cout << "Taking action" << std::endl;
        if (history.Size()) {
            if (history.LastAction() == 44) {
                return 60; //60 is Close gripper
            }
        }
        return 44;
    }
};

class GraspingObjectExplorationPolicy : public Policy {
protected:
	const GraspingV4* graspingV4_;
public:

    GraspingObjectExplorationPolicy(const DSPOMDP* model)
    : Policy(model),
        graspingV4_(static_cast<const GraspingV4*>(model)){
    }

    int Action(const std::vector<State*>& particles,
            RandomStreams& streams, History& history) const {
        // 44 is Move up in y with maximum value
        //std::cout << "Taking action" << std::endl;
        GraspingStateV4 *mostLikelyState = new GraspingStateV4(0,0,0,0,0);
        double max_weight = 0;
        for(int i = 0; i < particles.size(); i++)
        {
            if(particles[i]->weight > max_weight)
            {
                max_weight = particles[i]->weight; 
            }
        }
        int object_id[graspingV4_->num_sampled_objects];
        for(int  i = 0; i < graspingV4_->num_sampled_objects; i++)
        {
            object_id[i] = 0;
        }
        for(int i = 0; i < particles.size(); i++)
        {
            if(particles[i]->weight >= max_weight - 0.0001 )
            {
                GraspingStateV4 *grasping_state = static_cast<GraspingStateV4*> (particles[i]);
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
        for(int  i = 0; i < graspingV4_->num_sampled_objects; i++)
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
                if(mostLikelyState->x < 0 && mostLikelyState->x > graspingV4_->min_y_i)
                {
                    return graspingV4_->A_DECREASE_X + 14;
                }
                if(mostLikelyState->x > 0 && mostLikelyState->x < graspingV4_->max_y_i)
                {
                    return graspingV4_->A_INCREASE_X + 14;
                }
                return graspingV4_->A_DECREASE_Y + 14;
            }
            else
            {
                if(mostLikelyState->x + mostLikelyState->r < 0 || mostLikelyState->x - mostLikelyState->l > 0)
                {
                    return graspingV4_->A_DECREASE_Y + 14;
                }
                else
                {
                    return graspingV4_->A_CLOSE;
                }
                    
            }
        }
        else
        {
            if(mostLikelyState->x > -0.001 && mostLikelyState->x < 0.001)
            {
                return graspingV4_->A_INCREASE_Y + 14;
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
                    return graspingV4_->A_INCREASE_X + steps;
                }
                else
                {
                    return graspingV4_->A_DECREASE_X + steps;
                }
            }
        }
        
        return 44;
    }
};

void GraspingV4::InitializeScenarioLowerBound(RandomStreams& streams, std::string name) {
    if (name == "TRIVIAL" || name == "DEFAULT") {
        //the default lower bound from lower_bound.h
        scenario_lower_bound_ = new TrivialScenarioLowerBound(this);
    } else if (name == "UP") {
        //set EAST as the default action for each state
        scenario_lower_bound_ = new SimpleGraspingUpPolicy(this);
    }else if (name == "OE") {
        //set EAST as the default action for each state
        scenario_lower_bound_ = new GraspingObjectExplorationPolicy(this);
    } else {
        
        std::cerr << "GraspingV4:: Unsupported lower bound algorithm: " << name << std::endl;
        exit(0);
    }
}
 
 */