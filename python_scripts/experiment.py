

#for i in range(0,500):
   # print "./bin/despot --problem=graspingV3 -v 3 -t 120 --number=%d > 5_objects_obs_prob_change_particles_as_state/graspingV3_state_%d_t120_obs_prob_change_particles_as_state_5objects.log 2>&1" % (i,i)

#for i in range(0,500):
#    print "./bin/despot --problem=graspingV3 -v 3 --number=%d --solver=LEARNING --data-file=/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/all_logs.yaml > /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/learning/state_%d.log 2>&1" % (i,i)


#for i in range(0,500):
#    print "./bin/despot --problem=graspingV3 -v 3 --number=%d --solver=LEARNING --data-file=/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/all_logs.yaml > /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/learning/match_without_holes/only_one_match_without_reset/state_%d.log 2>&1" % (i,i)


#for i in range(0,500):
#    print "./bin/despot --problem=graspingV3 -v 3 --number=%d --solver=LEARNING --data-file=/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/all_logs.yaml > /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/learning/match_without_holes/match_along_length_without_reset/state_%d.log 2>&1" % (i,i)

#for i in range(0,400):
#    print "./bin/despot_model_version7_different_objects --problem=graspingV4 -v 3 -t 5 -n 10 --runs=4 --number=%d > 4_different_objects_obs_prob_change_particles_as_state/graspingV4_state_%d_multi_runs_t5_n10_obs_prob_change_particles_as_state_4objects.log 2>&1" % (i,i)

#for i in range(0,400):
#    print "./bin/despot --problem=graspingV4 -v3 --number=%d --solver=LEARNING --data-file=/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/all_logs.yaml >  /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/learning/state_%d.log 2>&1" % (i,i)

#for i in range(0,400):
#    print "./bin/despot --problem=graspingV4 -v3 --number=%d --solver=ADABOOST --data-file=/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/all_logs.yaml >  /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/adaboost_different_objects/state_%d.log 2>&1" % (i,i)

#for i in range(0,400):
#    print "./bin/despot --problem=graspingV4 -v3 --number=%d --solver=ADABOOST >  /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/adaboost_same_objects/state_%d.log 2>&1" % (i,i)

#for i in range(0,400):
#    print "./bin/despot --problem=graspingV4 -v3 --number=%d --solver=DEEPLEARNING --runs=4 >  /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version9/state_%d_multi_runs.log 2>&1" % (i,i)

#for i in range(0,400):
#    print "./bin/despot --problem=graspingV4 -v3 -t 10 -n 10 --runs=4 --number=%d --solver=LEARNINGPLANNING >  /home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_%d_t10_n10_multi_runs_obs_prob_change_particles_as_state_4objects.log 2>&1 " % (i,i)

#for i in range(0,400):
#    print "./bin/despot --problem=graspingV4 -v 3 -t 10 -n 10 --number=%d --belief=SINGLE_OBJECT > single_object_belief/training_objects/graspingV4_state_%d_t10_n10_obs_prob_change_single_object_radius_1_4objects_100_particles.log 2>&1" % (i,i)

#for i in range(0,100):
#    print "./bin/apc_despot_interface -v3 -t 5 -n 1 --number=%d -l CAP > ./results/despot_logs/VrepData_single_particle_belief_t5_n1_state_%d.log 2>&1" %(i,i)

for i in range(0,1000):
    print "./bin/apc_despot_interface -v3 -t 5 -n 10 --number=-1 -l CAP --belief=GAUSSIAN_WITH_STATE_IN > ./results/despot_logs/TableScene_gaussian_belief_with_state_in_belief_t5_n10_trial_%d.log 2>&1" %(i)

#for i in range(0,1000):
#    print "./bin/apc_despot_interface -v3 -t 1 --solver=DEEPLEARNING --number=%d -l CAP > ./despot_logs/learning_trial_%d.log 2>&1" %(i,i)
