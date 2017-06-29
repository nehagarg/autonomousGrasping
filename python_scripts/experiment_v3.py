
import os
import getopt
import sys
from plot_despot_results import get_list_input
import subprocess

    
initial_ros_port = 11311
max_ros_port = initial_ros_port + 50
running_nodes_to_screen = {}
running_screen_to_nodes = {}
stopped_nodes_to_screen = {}
stopped_screen_to_nodes = {}

def generate_despot_command(t, n, l, c, problem_type, pattern, begin_index, end_index, command_prefix):
    actual_command = ' python ../python_scripts/experiment_v2.py -e -p ' + problem_type
    actual_command = actual_command + ' -s ' + repr(begin_index)
    actual_command = actual_command + ' -c ' + repr(end_index)
    actual_command = actual_command + ' -t ' + t
    actual_command = actual_command + ' -n ' + n
    actual_command = actual_command + ' commands/' + command_prefix.replace('pattern', pattern)
    
    if c != 'None':
        actual_command = actual_command + '_combined_' + c
    elif l != 'None':
        actual_command = actual_command + '_learning'
    if l != 'empty' and l != 'None':    
        actual_command = actual_command + "_v" + l
    actual_command = actual_command + '.yaml'
    return actual_command
    
def generate_commands_file(file_name, problem_type, work_folder_dir, starting_screen_counter = 0, source_tensorflow = False, separate_ros_vrep_port = False):
    f = open(file_name, 'w')    
    global initial_ros_port
    global max_ros_port 
    starting_ros_port = initial_ros_port 
    vrep_ros_port = initial_ros_port + 1
    
    tensorflow_path = '~/tensorflow' #Assumed location of tensorflow dir
    vrep_dir =  work_folder_dir + '/V-REP_PRO_EDU_V3_3_2_64_Linux'
    problem_dir = work_folder_dir + "/neha_github/autonomousGrasping/" + problem_type
    if problem_type == 'despot_without_display':
        problem_dir = work_folder_dir + "/neha_github/autonomousGrasping/" + "/grasping_ros_mico"
    
    input_pattern = raw_input("Pattern type: all or file identifier?")
    pattern = input_pattern
    pattern_list = [pattern]
    if pattern == 'all':
        pattern_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    
    
    command_prefix = raw_input("Command prefix?")
    
    time_steps = ['1','5']
    time_steps = get_list_input(time_steps, "Planning times")

    sampled_scenarios = ['5', '10', '20', '40', '80', '160', '320', '640', '1280']
    sampled_scenarios = get_list_input(sampled_scenarios, "Sampled scenarios")


    learning_versions = ['8']
    learning_versions =  get_list_input(learning_versions, "Learning versions")


    combined_policy_versions = ['0', '1', '2']
    combined_policy_versions = get_list_input(combined_policy_versions, "Combined policy versions")

    begin_index = 0
    begin_index_input = raw_input("Begin index (default 0):")
    if begin_index_input:
        begin_index = int(begin_index_input)
    end_index = 1000
    end_index_input = raw_input("End index (default 1000):")
    if end_index_input:
        end_index = int(end_index_input)
    index_step = end_index
    index_step_input = raw_input("Index step (default " + repr(index_step) + " ):")
    if index_step_input:
        index_step = int(index_step_input)
        
   
    for l in learning_versions:
        for c in combined_policy_versions:
            for t in time_steps:
                for n in sampled_scenarios:
                    for p in pattern_list:
                      for b_index in range(begin_index, end_index, index_step):
                    
                        actual_command = 'cd ' + problem_dir + ';' + generate_despot_command(t, n, l, c, problem_type, p, b_index, b_index + index_step, command_prefix)
                        despot_screen_name = repr(starting_screen_counter)+ '_' + problem_type
                        if separate_ros_vrep_port:
                            starting_ros_port = vrep_ros_port
                            despot_screen_name = repr(starting_screen_counter)+ '_' + problem_type + '_' + repr(starting_ros_port)
                            roscore_screen_name = repr(starting_screen_counter)+ '_roscore_' + repr(starting_ros_port)
                            
                        f.write('screen -S ' + despot_screen_name + ' -d -m \n')
                        script_start_command = 'script ' + despot_screen_name
                        f.write("screen -S " + despot_screen_name + " -X stuff '" + script_start_command + " ^M' \n")
                            
                        if separate_ros_vrep_port:
                            ros_master_uri_command = 'export ROS_MASTER_URI=http://localhost:' +  repr(starting_ros_port)
                            roscore_command = 'roscore -p ' + repr(starting_ros_port)
                            f.write('screen -S ' + roscore_screen_name + ' -d -m  \n')
                            f.write("screen -S " + roscore_screen_name + " -X stuff '" + ros_master_uri_command +  " ^M' \n")
                            f.write("screen -S " + roscore_screen_name + " -X stuff '" + roscore_command +  " ^M' \n")
                            f.write("sleep 60s \n")
                            f.write("screen -S " + despot_screen_name + " -X stuff '" + ros_master_uri_command +  " ^M' \n")
                        
            
                            vrep_command = 'until rostopic list ; do sleep 1; done ; cd ' + vrep_dir + '; xvfb-run --auto-servernum --server-num=1 -s "-screen 0 640x480x24" ./vrep.sh -h ../vrep_scenes/micoWithSensorsMutliObjectTrialWithDespotIKVer4Cylinder' + p + '.ttt'
                            vrep_screen_name = repr(starting_screen_counter)+ '_vrep_' + repr(starting_ros_port)

                            f.write('screen -S ' +  vrep_screen_name + ' -d -m \n')
                            f.write("screen -S " + vrep_screen_name + " -X stuff '" + ros_master_uri_command +  " ^M' \n")
                            f.write("screen -S " + vrep_screen_name + " -X stuff '" + vrep_command +  " ^M' \n")
                            f.write("sleep 60s \n")
                            
                            actual_command = "until rostopic list | grep vrep ; do sleep 1; done ; " + actual_command
                            vrep_ros_port = vrep_ros_port + 1
                            if vrep_ros_port > max_ros_port:
                                vrep_ros_port = initial_ros_port + 1
                        if source_tensorflow:
                            tensorflow_command = 'source ' + tensorflow_path + '/bin/activate'
                            f.write("screen -S " + despot_screen_name + " -X stuff '" + tensorflow_command + " ^M'\n")
                            
                        f.write("screen -S " + despot_screen_name + " -X stuff '" + actual_command +  " ^M^D' \n")
                        starting_screen_counter = starting_screen_counter + 1

def get_screen_counter_from_command(command):
    screen_name = None
    screen_counter = None
    ros_port = None
    command_parts = command.split() #split on whitespaces
    if command_parts[0] == 'screen' and command_parts[1] == '-S':
        screen_name = command_parts[2]
    if screen_name is not None:
        (screen_counter, ros_port) = get_screen_counter_port_from_screen_name(screen_name)
    return (screen_name, screen_counter, ros_port)

def get_screen_counter_port_from_screen_name(screen_name):
    screen_counter = int(screen_name.split('_')[0])
    ros_port = -1
    try:
        ros_port = int(screen_name.split('_')[-1])
    except ValueError:
        pass
    return(screen_counter, ros_port)

def add_entry_to_running_nodes(running_nodes_to_screen, running_screen_to_nodes, node_name, screen_name):
    if node_name in running_nodes_to_screen:
        if screen_name not in running_nodes_to_screen[node_name]:
            running_nodes_to_screen[node_name].append(screen_name)
    else:
        running_nodes_to_screen[node_name] = [screen_name]
    if screen_name in running_screen_to_nodes.keys():
        assert(running_screen_to_nodes[screen_name] == node_name)
    else:
        running_screen_to_nodes[screen_name] = node_name
        
def update_running_nodes(running_node_file, running_nodes_to_screen, running_screen_to_nodes):
    ans = 0
    with open(running_node_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            node_name = values[0]
            screen_name = values[1]
            (screen_counter, ros_port) = get_screen_counter_port_from_screen_name(screen_name)
            if screen_counter > ans:
                ans = screen_counter
            add_entry_to_running_nodes(running_nodes_to_screen, running_screen_to_nodes, node_name, screen_name)
    return ans

def port_running_on_node(screen_port, node):
    global running_nodes_to_screen   
    global stopped_nodes_to_screen
    if node in running_nodes_to_screen.keys():
        screen_name_list = list(running_nodes_to_screen[node])
        if node in stopped_nodes_to_screen.keys():
            for screen_name in stopped_nodes_to_screen[node]:
                screen_name_list.remove(screen_name)
        for screen_name in screen_name_list:
            (screen_counter, ros_port) = get_screen_counter_port_from_screen_name(screen_name)
            if screen_port == ros_port:
                return True
    return False 
def get_maximum_load_for_node(node):
    if 'ncl' in node:
        return 20
    if 'eagle' in node:
        return 28
    return 4
    
def assign_node(node_list, screen_name, running_node_file):
    global initial_ros_port
    global running_nodes_to_screen 
    global running_screen_to_nodes 
    global stopped_nodes_to_screen 
    global stopped_screen_to_nodes 
    
    (screen_counter, screen_port) = get_screen_counter_port_from_screen_name(screen_name)
    
    for node in node_list:
        #check node ssh
        command = "timeout 5 ssh " + node + " echo 'hello'"
        success = run_command_on_node(command )
        if success is None:
            continue
        #check node load
        command = "ssh " + node + " cat /proc/loadavg | awk '{print $1}'"
        avg_load = 100
        try:
            output = run_command_on_node(command )
            avg_load = float(output)
        except ValueError:
            print output
            print "Taking default avg load of " + repr(avg_load)
            
        max_av_load = get_maximum_load_for_node(node)
        if avg_load > max_av_load:
            continue
        
        #if vrep node (screen_port is > initial_ros_port) node check in the file containing vrep ports and nodes
        if screen_port > initial_ros_port:
            if port_running_on_node(screen_port, node):
                continue
        elif 'despot_without_display' in screen_name:
            do_roscore_setup([node])
        
        #update ans
        #update file containing screen_counters and node
        add_entry_to_running_nodes(running_nodes_to_screen, running_screen_to_nodes, node, screen_name)
        with open(running_node_file, 'a' ) as f:
            f.write(node + " " + screen_name + "\n")
        
        #assign node
        return node
    
    return None   
    

def run_command_on_node(command, node = None):
    if node is not None: 
        command = 'ssh ' + node + ' "' + command.replace('"', '\\"') + ' "'
    ans = None
    try:
        print "Executing : " + command
        ans = subprocess.check_output(["bash", "-O", "extglob", "-c", command])
    except subprocess.CalledProcessError as e:
        print "Caught Called Process Error"
    return ans


def check_finished_processes(stopped_node_file):
    global initial_ros_port
    global stopped_nodes_to_screen 
    global stopped_screen_to_nodes
    global running_nodes_to_screen 
    global running_screen_to_nodes
    for screen_name in running_screen_to_nodes.keys():
        if screen_name in stopped_screen_to_nodes.keys():
            continue
        node_name = running_screen_to_nodes[screen_name]
        #try stopping the screen process
        command = "screen -S "  + screen_name + " -X stuff '^D'"
        run_command_on_node(command, node_name)
        output = run_command_on_node(command, node_name)
        if output is None: #screen stopped
            (screen_counter, ros_port) = get_screen_counter_port_from_screen_name(screen_name)
            if ros_port > initial_ros_port :
                #stop vrep and roscore screens
                vrep_screen_name = '_'.join([repr(screen_counter), "vrep", repr(ros_port)])
                roscore_screen_name = '_'.join([repr(screen_counter), "roscore", repr(ros_port)])
                command =  "screen -S "  + vrep_screen_name + " -X stuff '^C'"
                run_command_on_node(command, node_name)
                command = "screen -S "  + vrep_screen_name + " -X stuff '^D'"
                run_command_on_node(command, node_name)
                output = run_command_on_node(command, node_name)
                if output is None:
                    command = "screen -S "  + roscore_screen_name + " -X stuff '^C'"
                    run_command_on_node(command, node_name)
                    command = "screen -S "  + roscore_screen_name + " -X stuff '^D'"
                    run_command_on_node(command, node_name)
                    output = run_command_on_node(command, node_name)
            if output is None:        
                with open(stopped_node_file, 'a' ) as f:
                    f.write(node_name + " " + screen_name + "\n")

                add_entry_to_running_nodes(stopped_nodes_to_screen, stopped_screen_to_nodes, node_name, screen_name)    
                
def update_nodes(node_file_name):        
    with open(node_file_name, 'r') as f:
        nodes = f.readlines()
    nodes = [x.strip() for x in nodes]
    return nodes
        
def do_roscore_setup(nodes):
    for node in nodes:    
        output = run_command_on_node('screen -S roscore -X select .', node) #check if screen exists
        if output is None: #command unsuccessful screen does not exist
            run_command_on_node('screen -S roscore -d -m', node) #create screen
            run_command_on_node("screen -S roscore -X stuff 'roscore^M'", node) #start roscore
            run_command_on_node("sleep 60s")
        else: #screen already exists
            #check if roscore running
            run_command_on_node("screen -S roscore -X stuff '^D'", node)
            output = run_command_on_node("screen -S roscore -X stuff '^D'", node)
            if output is None: #roscore not running and screen killed
                run_command_on_node('screen -S roscore -d -m', node) #create screen
                run_command_on_node("screen -S roscore -X stuff 'roscore^M'", node) #start roscore
                run_command_on_node("sleep 60s")

def next_screen_counter(screen_counter_list, current_screen_counter):
    if screen_counter_list is None:
        return current_screen_counter + 1
    else:
        if current_screen_counter < screen_counter_list[0]
            return screen_counter_list[0]
        if current_screen_counter >= screen_counter_list[-1]:
            return 10000000  #Something more than number of commands in file
        else:
            i = screen_counter_list.index(current_screen_counter) + 1
            if i < len(screen_counter_list):
                return screen_counter_list[i]
                
            
def all_processes_stopped():
    global running_screen_to_nodes
    global stopped_screen_to_nodes
    running_screens = running_screen_to_nodes.keys()
    stopped_screens = stopped_screen_to_nodes.keys()
    
    return set(running_screens) == set(stopped_screens)
    
def run_command_file(command_file_name, node_file_name, running_node_file, stopped_node_file, current_screen_counter_file, screen_counter_list_file):    
    nodes = update_nodes(node_file_name)
    
    #update ports on each node
    global running_nodes_to_screen 
    global running_screen_to_nodes 
    global stopped_nodes_to_screen 
    global stopped_screen_to_nodes
    start_screen_counter = update_running_nodes(running_node_file, running_nodes_to_screen, running_screen_to_nodes)
    update_running_nodes(stopped_node_file, stopped_nodes_to_screen, stopped_screen_to_nodes)
    
    start_screen_counter_list = None
    if screen_counter_list_file is not None:
        print "Reading counter list from " + screen_counter_list_file
        with open(screen_counter_list_file, 'r') as f:
            all_lines = f.readlines()
            start_screen_counter_list = sorted(set([int(x) for x in all_lines]))
        
    with open(current_screen_counter_file, 'r') as f:
        assert(int(f.readline() ) ==   start_screen_counter)
    
    existing_screen_counter = start_screen_counter
    assigned_node = None
    line_number_found = False
    with open(command_file_name) as f:
        for line in f:
            command = line.strip()
            (screen_name, screen_counter, screen_port) = get_screen_counter_from_command(command)
            if not line_number_found:
                #print screen_counter
                if screen_counter != next_screen_counter(start_screen_counter_list, existing_screen_counter):
                    continue
                else:
                    line_number_found = True
            if screen_counter is None:
                print "Executing " + command
                os.system(command)
            else:
                if screen_counter != existing_screen_counter:
                    with open(current_screen_counter_file, 'w') as f:
                        f.write(repr(existing_screen_counter))  
                    if screen_counter != next_screen_counter(start_screen_counter_list, existing_screen_counter):
                        continue
                    existing_screen_counter = screen_counter
                    assigned_node = None
                    while assigned_node is None:
                        assigned_node = assign_node(nodes, screen_name, running_node_file)
                        if assigned_node is None:
                            print "All nodes busy. Sleeping..."
                            run_command_on_node('sleep 300s')
                            nodes = update_nodes(node_file_name)
                            check_finished_processes(stopped_node_file)


                    
                else:
                    assert(assigned_node is not None)
                #not checking if a screen with a given name exists on the node, assign node will take care of it
                output = run_command_on_node(command, assigned_node)
                if output is None:
                    print "Command not executed. Exiting"
                    sys.exit()
            
            
    with open(current_screen_counter_file, 'w') as f:
            f.write(repr(existing_screen_counter))      
    
    #screen -S Jetty -X kill ; echo $?
    #screen -S Jetty -X stuff '^D'



def main():
    opts, args = getopt.getopt(sys.argv[1:],"hegvtd:p:s:",["dir="])
    work_folder_dir = None
    command_file = None
    execute_command_file = False
    genarate_command_file = False
    separate_ros_vrep_port = False
    source_tensorflow = False
    starting_screen_counter = 1

    problem_type = None
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'experiment_v3.py -e |-g -v -t  -s starting_screen_counter -p problem_type -d work_folder_dir command_file'
         sys.exit()
      elif opt == '-e':
         execute_command_file = True
      elif opt == '-g':
         genarate_command_file = True
      elif opt == '-v':
         separate_ros_vrep_port = True
      elif opt == '-t':
         source_tensorflow = True
      elif opt == '-p':
          problem_type = arg
      elif opt == '-s':
          starting_screen_counter = int(arg)
      elif opt in ("-d", "--dir"):
         work_folder_dir = arg   
         
    if len(args) > 0:
        command_file = args[0]
        
    if genarate_command_file:
        generate_commands_file(command_file, problem_type, work_folder_dir,  starting_screen_counter, source_tensorflow, separate_ros_vrep_port)
        
    
    if execute_command_file:
        current_screen_counter_file = "run_txt_files/current_screen_counter.txt"
        current_screen_counter = 0
        if not os.path.exists(current_screen_counter_file):
            with open(current_screen_counter_file, 'w') as f:
                f.write(repr(current_screen_counter))
        running_nodes_file = "run_txt_files/running_nodes.txt"
        if not os.path.exists(running_nodes_file):
            os.system('touch ' + running_nodes_file)
        stopped_nodes_file = "run_txt_files/stopped_nodes.txt"
        if not os.path.exists(stopped_nodes_file):
            os.system('touch ' + stopped_nodes_file)
        counter_list_file_name = "run_txt_files/screen_counter_list.txt"
        counter_list_file = None
        if os.path.exists(counter_list_file_name):
            counter_list_file = counter_list_file_name
            
        
        while True:            
            run_command_file(command_file, "run_txt_files/node_list.txt",running_nodes_file, stopped_nodes_file , current_screen_counter_file, counter_list_file)
            with open(current_screen_counter_file, 'r') as f:
                new_screen_counter = int(f.readline())
                if new_screen_counter == current_screen_counter:
                    break
                else:
                    current_screen_counter = new_screen_counter
                    #TODO automatically merge runnign and stopped nodes files using command
                    #awk 'NR==FNR{a[$0];next} !($0 in a)' test_stopped_nodes.txt test_running_nodes.txt
        while(not all_processes_stopped()):
            print "Sleeping before checking process status..."
            run_command_on_node('sleep 600s')
            check_finished_processes(stopped_nodes_file)
        
if __name__ == '__main__':
    main()
    