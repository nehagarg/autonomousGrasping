import re
import os
import getopt
import sys
#from plot_despot_results import get_list_input
import subprocess
from grasping_object_list import get_grasping_object_name_list

def make_object_dirs(dir_name, object_group):
    object_list = get_grasping_object_name_list(object_group)
    for object in object_list:
        os.mkdir(dir_name + "/" + object)

initial_ros_port = 11311
max_ros_port = initial_ros_port + 50
current_ros_port = initial_ros_port
running_nodes_to_screen = {}
running_screen_to_nodes = {}
stopped_nodes_to_screen = {}
stopped_screen_to_nodes = {}
last_assigned_node = None
vrep_scene_version = "7"
generic_scene = False
port_name_macro = 'PORT_NAME'
assigned_port = None #Port assigned if true port is already occupied on node
true_port = None #Intended port
true_GPU = None #Intended GPU
assigned_GPU = None #GPU assigned if true port is already occupied on node
num_gpus = 4
#Copied from plot_despot_results to avoid importing the modeules it is dependent on
def get_list_input(sampled_scenarios, command):
    while True:
        input = raw_input(command + " are " + " ".join(map(str, sampled_scenarios)) + " To add type a <no>. To remove type r <no>. To stop type s.")
        if 's' in input:
            break
        if 'a' in input:
            sampled_scenarios.append(input.split(' ')[1])
            sampled_scenarios = sorted(set(sampled_scenarios))
        if 'r' in input:
            sampled_scenarios.remove(input.split(' ')[1])
    return sampled_scenarios


def get_gather_data_number(pattern, t):
    if 'G3DB' in pattern:
        num = int(pattern.split('_')[0][4:])
        return ((1000+num)*10) + int(t)
    else:
        return (int(filter(str.isdigit, pattern))*10) + int(t)

def generate_despot_command(t, n, l, c, problem_type, pattern, begin_index, end_index, command_prefix,gpu_count):
    if(problem_type.startswith('despot_bt')):
        command = "./" + command_prefix + " -v 3 "
        command = command + "--solver=" + pattern + " "
        if (t != 'None'):
            command = command + "-t " + t + " "
        if (n != 'None'):
            command = command + "-n " + n + " "
        command = command + " > ../../../../examples/cpp_models/" + problem_type.split('_')[-1]
        command = command + "/results/" + command_prefix +  "/t" + t + "_n" + n + "/"
        command = command + pattern + "_trial_"  + repr(begin_index) + ".log 2>&1"
        return command


    if(command_prefix == 'label_g3db_objects'):
        # python label_g3db_objects.py -o ../grasping_ros_mico/g3db_object_labels/ ../../../vrep/G3DB_object_dataset/obj_files/
        #python label_g3db_objects.py -o ../grasping_ros_mico/pure_shape_labels all_cylinders
        #pattern can be g3db_object_labels or pure_shape_labels
        command = 'python label_g3db_objects.py -u -o ../grasping_ros_mico/'
        command = command + pattern
        if(pattern == 'pure_shape_labels'):
            command = command + ' all_cylinders'
        else:
            command = command + ' ' + t + "/" + repr(begin_index) + '_'
        return command
    if('generate_point_clouds' in command_prefix):
        # python label_g3db_objects.py -o ../grasping_ros_mico/g3db_object_labels/object_instances/object_instances_updated ../../../vrep/G3DB_object_dataset/obj_files/
        #python label_g3db_objects.py -o ../grasping_ros_mico/pure_shape_labels all_cylinders
        #pattern can be g3db_object_labels or pure_shape_labels
        command_part = '-p'
        if 'classification' in command_prefix:
            command_part = '-q'
        command = 'python label_g3db_objects.py ' + command_part + ' -o ../grasping_ros_mico/'
        command = command + pattern
        if(pattern == 'pure_shape_labels'):
            command = command + ' all_cylinders'
        else:
            command = command + ' ' + t + "/" + repr(begin_index) + '_'
        return command


    if('gather_data' in command_prefix):
        #num = get_gather_data_number(pattern, t)
        command = './bin/gather_data ' + pattern + ' ' + t + ' ' + n
        command_prefix_parts = command_prefix.split('_')
        y_start = -1
        y_end = -1
        rot_z = "-1"
        if len(command_prefix_parts) > 2:
            y_start = int(command_prefix_parts[2])
            y_end = y_start + 1

        command = command + ' ' + ",".join(map(str,[begin_index, end_index, y_start, y_end]))
        command = command + ' ' + l
        #if len(command_prefix_parts) > 3:
        #    rot_z = command_prefix_parts[3]
        #command = command + ' ' + rot_z
        if c != 'None':
            command = command + ' ' + c
        return command
    if('grasping_dynamic_model' in command_prefix):
        [classifier_type, task_type] = t.split('%')
        out_file_name = 'data_for_regression/' + pattern + "/"
        out_file_name = out_file_name + classifier_type + "-" + task_type + ".log"
        command = 'python scripts/grasping_dynamic_model.py '
        command = command + '-c ' + classifier_type
        command = command + ' -t ' + task_type
        command = command + ' -o ' + pattern
        command = command + ' -d data_for_regression > ' + out_file_name
        return command
    actual_command = ' python ../python_scripts/experiment_v2.py -e -p ' + problem_type
    if problem_type == 'grasping_with_vision':
        actual_command = ' python python_scripts/experiment_v2.py -e -p ' + problem_type
    actual_command = actual_command + ' -s ' + repr(begin_index)
    actual_command = actual_command + ' -c ' + repr(end_index)
    actual_command = actual_command + ' -t ' + t
    actual_command = actual_command + ' -n ' + n
    if problem_type == 'grasping_with_vision':
        actual_command = actual_command + ' -j ' + repr(gpu_count)
    actual_command = actual_command + ' commands/' + command_prefix.replace('pattern', pattern)

    if c != 'None':
        actual_command = actual_command + '_combined_' + c
    elif l != 'None':
        actual_command = actual_command + '_learning'
    if l != 'empty' and l != 'None':
        actual_command = actual_command + "_v" + l
    actual_command = actual_command + '.yaml'
    return actual_command

def generate_commands_file(file_name, problem_type, work_folder_dir, starting_screen_counter = 1, tensorflow_path_input = '', separate_ros_vrep_port = False, command_list_file = None):
    f = open(file_name, 'w')
    global initial_ros_port
    global max_ros_port
    global vrep_scene_version
    global generic_scene
    global current_ros_port
    global num_gpus
    starting_ros_port = initial_ros_port
    vrep_ros_port = current_ros_port + 1

    tensorflow_path = tensorflow_path_input #Assumed location of tensorflow dir
    source_tensorflow = tensorflow_path != ''
    vrep_dir =  work_folder_dir + '/V-REP_PRO_EDU_V3_3_2_64_Linux'
    problem_dir = work_folder_dir + "/neha_github/autonomousGrasping/" + problem_type
    if problem_type == 'despot_without_display':
        problem_dir = work_folder_dir + "/neha_github/autonomousGrasping/" + "/grasping_ros_mico"
    if problem_type.startswith('despot_bt'):
        problem_dir = work_folder_dir + "/neha_github/despot_bt/build/examples/cpp_models/"
        problem_dir = problem_dir + problem_type.split('_')[-1]
    if problem_type == 'grasping_with_vision':
        problem_dir =  work_folder_dir + "/AdaCompNUS/HyP-Despot-LargeObservation-clone-neha-laptop/src/HyP_examples/grasping_ros_mico"

    if command_list_file is not None:
        with open(command_list_file, 'r') as ff:
            a =  ff.readlines()
            inputs = a[0].split(' ')
    if command_list_file is None:
        input_pattern = raw_input("Pattern type: all or file identifier?")
    else:
        input_pattern = inputs[0]
    pattern = input_pattern
    #pattern_list = [pattern]
    if pattern == 'all':
        pattern_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    elif pattern == 'grasp_objects':
        pattern_list = get_grasping_object_name_list()
    else:
        pattern_list = get_grasping_object_name_list(pattern)

    if command_list_file is None:
        command_prefix = raw_input("Command prefix?")
    else:
        command_prefix = inputs[1]

    time_steps = ['1','5']
    sampled_scenarios = ['5', '10', '20', '40', '80', '160', '320', '640', '1280']
    learning_versions = ['8']
    combined_policy_versions = ['0', '1', '2']
    begin_index = 0
    end_index = 1000
    if command_list_file is None:
        time_steps = get_list_input(time_steps, "Planning times")
        sampled_scenarios = get_list_input(sampled_scenarios, "Sampled scenarios")
        learning_versions =  get_list_input(learning_versions, "Learning versions")
        combined_policy_versions = get_list_input(combined_policy_versions, "Combined policy versions")
        begin_index_input = raw_input("Begin index (default 0):")
        end_index_input = raw_input("End index (default 1000):")
    else:
        time_steps = inputs[2].split(',')
        sampled_scenarios = inputs[3].split(',')
        learning_versions =  inputs[4].split(',')
        combined_policy_versions = inputs[5].split(',')
        begin_index_input = inputs[6]
        end_index_input = inputs[7]




    if begin_index_input:
        begin_index = int(begin_index_input)


    if end_index_input:
        end_index = int(end_index_input)
    index_step = end_index

    if command_list_file is None:
        index_step_input = raw_input("Index step (default " + repr(index_step) + " ):")
    else:
        index_step_input = inputs[8]

    if index_step_input:
        index_step = int(index_step_input)

    for l in learning_versions:
        for c in combined_policy_versions:
            for t in time_steps:
                for n in sampled_scenarios:
                    for p in pattern_list:
                      for b_index in range(begin_index, end_index, index_step):
                        gpu_count = (starting_screen_counter) % num_gpus
                        actual_command = 'cd ' + problem_dir + ';' + generate_despot_command(t, n, l, c, problem_type, p, b_index, b_index + index_step, command_prefix,gpu_count)
                        gpu_screen_name = ''
                        if problem_type ==  'grasping_with_vision':
                            gpu_screen_name = repr(gpu_count) + '_'
                        despot_screen_name = repr(starting_screen_counter)+ '_' +gpu_screen_name +  problem_type
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
                            f.write("sleep 1 \n")
                            f.write("screen -S " + despot_screen_name + " -X stuff '" + ros_master_uri_command +  " ^M' \n")


                            vrep_command = 'until rostopic list ; do sleep 1; done ; cd '
                            vrep_command = vrep_command + vrep_dir
                            vrep_command = vrep_command + '; xvfb-run --auto-servernum --server-num=1 -s "-screen 0 640x480x24" ./vrep.sh -h '
                            vrep_command = vrep_command + '../vrep_scenes/micoWithSensorsMutliObjectTrialWithDespotIKVer' + vrep_scene_version

                            if generic_scene:
                               vrep_command = vrep_command + 'G3DB_generic.ttt'
                            else:
                                if 'G3DB' not in p:
                                    vrep_command = vrep_command +  'Cylinder'
                                vrep_command = vrep_command + p + '.ttt'


                            vrep_screen_name = repr(starting_screen_counter)+ '_vrep_' + repr(starting_ros_port)

                            f.write('screen -S ' +  vrep_screen_name + ' -d -m \n')
                            f.write("screen -S " + vrep_screen_name + " -X stuff '" + ros_master_uri_command +  " ^M' \n")
                            f.write("screen -S " + vrep_screen_name + " -X stuff '" + vrep_command +  " ^M' \n")
                            f.write("sleep 1 \n")

                            actual_command = "until rostopic list | grep vrep ; do sleep 1; done ; " + actual_command
                            vrep_ros_port = vrep_ros_port + 1
                            if vrep_ros_port > max_ros_port:
                                vrep_ros_port = initial_ros_port + 1
                        if source_tensorflow:
                            tensorflow_command = 'source ' + tensorflow_path + '/bin/activate'
                            f.write("screen -S " + despot_screen_name + " -X stuff '" + tensorflow_command + " ^M'\n")

                        f.write("screen -S " + despot_screen_name + " -X stuff '" + actual_command +  " ^M^D' \n")
                        if problem_type.startswith('despot_bt'):
                            f.write("sleep 1 \n") #Required because sometimes same seeds are generated

                        starting_screen_counter = starting_screen_counter + 1
    return starting_screen_counter

def get_screen_counter_from_command(command):
    screen_name = None
    screen_counter = None
    ros_port = None
    command_parts = command.split() #split on whitespaces
    if command_parts[0] == 'screen' and command_parts[1] == '-S':
        screen_name = command_parts[2]
    if screen_name is not None:
        (screen_counter, ros_port,gpu_port) = get_screen_counter_port_from_screen_name(screen_name)
    return (screen_name, screen_counter, ros_port, gpu_port)

def get_screen_counter_port_from_screen_name(screen_name):
    screen_counter = int(screen_name.split('_')[0])
    gpu_port = -1

    try:
        gpu_port = int(screen_name.split('_')[1])
    except ValueError:
        pass
    ros_port = -1
    try:
        ros_port = int(screen_name.split('_')[-1])
    except ValueError:
        pass
    return(screen_counter, ros_port, gpu_port)

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
            (screen_counter, ros_port, gpu_port) = get_screen_counter_port_from_screen_name(screen_name)
            if screen_counter > ans:
                ans = screen_counter
            add_entry_to_running_nodes(running_nodes_to_screen, running_screen_to_nodes, node_name, screen_name)
    return ans

def GPU_running_on_node(screen_GPU, node):
    return port_running_on_node(screen_GPU, node, True)

def port_running_on_node(screen_port, node, check_gpu_port = False):
    global running_nodes_to_screen
    global stopped_nodes_to_screen
    if node in running_nodes_to_screen.keys():
        screen_name_list = list(running_nodes_to_screen[node])
        if node in stopped_nodes_to_screen.keys():
            for screen_name in stopped_nodes_to_screen[node]:
                screen_name_list.remove(screen_name)
        for screen_name in screen_name_list:
            (screen_counter, ros_port,gpu_port) = get_screen_counter_port_from_screen_name(screen_name)
            if(check_gpu_port):
                if(screen_port == gpu_port):
                    return True
                else:
                    if screen_port == ros_port:
                        return True
    return False
def get_maximum_load_for_node(node):
    if 'ncl' in node:
        return 20
    if 'eagle' in node:
        return 28
    if 'unicorn0' in node:
        return 28
    return 4

def assign_node(node_list, screen_name, running_node_file):

    global initial_ros_port
    global max_ros_port
    global running_nodes_to_screen
    global running_screen_to_nodes
    global stopped_nodes_to_screen
    global stopped_screen_to_nodes
    global last_assigned_node
    global assigned_port
    global true_port
    global num_gpus
    global true_GPU
    global assigned_GPU

    true_port = None
    assigned_port = None
    (screen_counter, screen_port, screen_GPU) = get_screen_counter_port_from_screen_name(screen_name)
    true_GPU = None
    assigned_GPU = None


    node_start_index = 0
    if last_assigned_node is not None:
        node_start_index = node_list.index(last_assigned_node)

    for node_index in range(0,len(node_list)):
        node = node_list[(node_index +node_start_index) % len(node_list) ]
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
            true_port = screen_port
            if port_running_on_node(screen_port, node):
                for possible_port in range(initial_ros_port+1, max_ros_port ): #Search for available port
                    if not port_running_on_node(possible_port, node):
                        assigned_port = possible_port
                        break
                if assigned_port is None:
                    continue;
        elif 'despot_without_display' in screen_name or 'grasping_with_vision' in screen_name:
            true_port = None
            do_roscore_setup([node])

        if screen_GPU >= 0:
            true_GPU = screen_GPU
            if GPU_running_on_node(screen_GPU, node):
                for possible_GPU in range(0,num_gpus): #Search for a free GPU
                    if not GPU_running_on_node(possible_GPU, node):
                        assigned_GPU = possible_GPU
                        break
                if assigned_GPU is None:
                    continue

        #update ans
        #update file containing screen_counters and node
        new_screen_name = screen_name
        if true_port is not None and assigned_port is not None:
            new_screen_name = re.sub(repr(true_port)+'$', repr(assigned_port), screen_name)
        if true_GPU is not None and assigned_GPU is not None:
            new_screen_name = re.sub('_'+repr(true_GPU)+ '_', '_'+repr(assigned_GPU)+ '_', screen_name)

        add_entry_to_running_nodes(running_nodes_to_screen, running_screen_to_nodes, node, new_screen_name)
        with open(running_node_file, 'a' ) as f:
            f.write(node + " " + new_screen_name + "\n")

        #assign node
        last_assigned_node = node
        return node

    return None


def run_command_on_node(command, node = None):
    global true_port
    global assigned_port
    global true_GPU
    global assigned_GPU

    if node is not None:
        if true_port is not None and assigned_port is not None:
            command = command.replace(repr(true_port) + ' ', repr(assigned_port) + ' ')
        if true_GPU is not None and assigned_GPU is not None:
            command = command.replace('-j ' + repr(true_GPU) + ' ', '-j ' + repr(assigned_GPU) + ' ')
            command = command.replace('_'+repr(true_GPU)+ '_', '_'+repr(assigned_GPU)+ '_')
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
    for screen_name in sorted(running_screen_to_nodes.keys()):
        if screen_name in stopped_screen_to_nodes.keys():
            continue
        node_name = running_screen_to_nodes[screen_name]
        #check node ssh
        command = "timeout 5 ssh " + node_name + " echo 'hello'"
        success = run_command_on_node(command )
        if success is None:
            print "Could not ssh. So not updated finished process list"
            continue
        #try stopping the screen process
        command = "screen -S "  + screen_name + " -X stuff '^D'"
        run_command_on_node(command, node_name)
        output = run_command_on_node(command, node_name)
        if output is None: #screen stopped
            (screen_counter, ros_port,gpu_port) = get_screen_counter_port_from_screen_name(screen_name)
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

def kill_roscore(node_file_name):
    nodes = update_nodes(node_file_name)
    all_nodes_free = False
    while(not all_nodes_free):
        all_nodes_free = True
        for node in nodes:
            output = run_command_on_node('screen -S roscore -X select .', node) #check if screen exists
            if output is not None: #command successful screen exists
                all_nodes_free = False
                run_command_on_node("screen -S roscore -X stuff '^C'", node)
                run_command_on_node("screen -S roscore -X stuff '^D'", node)
                run_command_on_node("screen -S roscore -X stuff '^D'", node)

def do_roscore_setup(nodes):
    for node in nodes:
        output = run_command_on_node('screen -S roscore -X select .', node) #check if screen exists
        if output is None: #command unsuccessful screen does not exist
            run_command_on_node('screen -S roscore -d -m', node) #create screen
            run_command_on_node("screen -S roscore -X stuff 'roscore^M'", node) #start roscore
            run_command_on_node("sleep 60")
        else: #screen already exists
            #check if roscore running
            run_command_on_node("screen -S roscore -X stuff '^D'", node)
            output = run_command_on_node("screen -S roscore -X stuff '^D'", node)
            if output is None: #roscore not running and screen killed
                run_command_on_node('screen -S roscore -d -m', node) #create screen
                run_command_on_node("screen -S roscore -X stuff 'roscore^M'", node) #start roscore
                run_command_on_node("sleep 60")

def next_screen_counter(screen_counter_list, current_screen_counter):
    if screen_counter_list is None:
        return current_screen_counter + 1
    else:
        if current_screen_counter < screen_counter_list[0]:
            return screen_counter_list[0]
        if current_screen_counter >= screen_counter_list[-1]:
            return current_screen_counter + 1  #Assuming we want to start with the rest of the commands
        else:
            i = screen_counter_list.index(current_screen_counter) + 1
            return screen_counter_list[i]


def all_processes_stopped():
    global running_screen_to_nodes
    global stopped_screen_to_nodes
    running_screens = running_screen_to_nodes.keys()
    stopped_screens = stopped_screen_to_nodes.keys()

    return set(running_screens) == set(stopped_screens)

def run_command_file(command_file_name, node_file_name, running_node_file, stopped_node_file, current_screen_counter_file, screen_counter_list_file, force_counter):
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
        existing_screen_counter = int(f.readline() )
        if not force_counter:
            assert( existing_screen_counter ==   start_screen_counter)

    #existing_screen_counter = start_screen_counter
    assigned_node = None
    line_number_found = False
    with open(command_file_name) as f:
        for line in f:
            command = line.strip()
            (screen_name, screen_counter, screen_port, gpu_port) = get_screen_counter_from_command(command)
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
                        line_number_found = False
                        continue
                    existing_screen_counter = screen_counter
                    assigned_node = None
                    while assigned_node is None:
                        assigned_node = assign_node(nodes, screen_name, running_node_file)
                        if assigned_node is None:
                            print "All nodes busy. Sleeping..."
                            run_command_on_node('sleep 300')
                            nodes = update_nodes(node_file_name)
                            #Will be done by a different process
                            #check_finished_processes(stopped_node_file)
                            update_running_nodes(stopped_node_file, stopped_nodes_to_screen, stopped_screen_to_nodes)



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

def check_finished_processes_standalone(running_node_file, stopped_node_file):
   #update ports on each node
    global running_nodes_to_screen
    global running_screen_to_nodes
    global stopped_nodes_to_screen
    global stopped_screen_to_nodes
    update_running_nodes(running_node_file, running_nodes_to_screen, running_screen_to_nodes)
    update_running_nodes(stopped_node_file, stopped_nodes_to_screen, stopped_screen_to_nodes)
    while(not all_processes_stopped()):
            run_command_on_node('sleep 10') #To avoid newly started process being stopped

            check_finished_processes(stopped_node_file)
            print "Sleeping before checking process status..."
            run_command_on_node('sleep 360')
            update_running_nodes(running_node_file, running_nodes_to_screen, running_screen_to_nodes)
            #update_running_nodes(stopped_node_file, stopped_nodes_to_screen, stopped_screen_to_nodes)

def generate_error_re_run_commands(command_file, problem_type, work_folder_dir,  starting_screen_counter, source_tensorflow, separate_ros_vrep_port, command_list_file):
    command = "vrep_model_fixed_distribution_multi_object_pattern_low_friction"

    input_text = None
    if os.path.exists(command_list_file):
        with open(command_list_file, 'r') as ff:
            input_text =  ff.readlines()
    if input_text is not None:
        inputs = [x.split(' ') for x in input_text]
    else:
        inputs = [['7cm', command, '5', '80', 'None', '1', '215', '216', '1']]
        inputs.append(['8cm', command, '5', '80', 'None', '1', '207', '208', '1'])
        inputs.append(['9cm', command, '5', '80', 'None', '1', '200', '201', '1'])
        inputs.append(['7cm', command, '5', '160', 'None', '1', '188', '189', '1'])
        inputs.append(['9cm', command, '5', '640', 'None', '1', '20', '21', '1'])
        inputs.append(['9cm', command, '5', '1280', 'None', '1', '227', '228', '1'])
        inputs.append(['8cm', command, '1', '10', 'None', '2', '223', '224', '1'])
        inputs.append(['9cm', command, '1', '10', 'None', '2', '238', '239', '1'])
        inputs.append(['8cm', command, '1', '20', 'None', '2', '232', '233', '1'])
        inputs.append(['8cm', command, '1', '40', 'None', '2', '64', '65', '1'])
        inputs.append(['9cm', command, '1', '80', 'None', '2', '114', '115', '1'])
        #inputs.append(['9cm', command, '1', '320', 'None', '2', '162', '163', '1'])
        inputs.append(['9cm', command, '1', '640', 'None', '2', '153', '154', '1'])
        inputs.append(['8cm', command, '1', '640', 'None', '2', '177', '178', '1'])
        inputs.append(['7cm', command, '1', '640', 'None', '2', '146', '147', '1'])
        inputs.append(['8cm', command, '1', '1280', 'None', '2', '112', '113', '1'])
        inputs.append(['8cm', command, '5', '5', 'None', '2', '75', '76', '1'])
        inputs.append(['8cm', command, '5', '10', 'None', '2', '25', '26', '1'])
        inputs.append(['8cm', command, '5', '20', 'None', '2', '17', '18', '1'])
        inputs.append(['8cm', command, '5', '40', 'None', '2', '30', '31', '1'])
        inputs.append(['9cm', command, '5', '40', 'None', '2', '21', '22', '1'])
        inputs.append(['8cm', command, '5', '80', 'None', '2', '22', '23', '1'])
        inputs.append(['85mm', command, '5', '5', 'None', '2', '30', '31', '1'])
        inputs.append(['75mm', command, '5', '10', 'None', '2', '16', '17', '1'])
        inputs.append(['85mm', command, '1', '1280', 'None', '2', '53', '54', '1'])
        inputs.append(['75mm', command, '1', '320', 'None', '2', '181', '182', '1'])
        inputs.append(['75mm', command, '1', '160', 'None', '2', '178', '179', '1'])
        inputs.append(['75mm', command, '1', '20', 'None', '2', '56', '57', '1'])
        inputs.append(['85mm', command, '1', '10', 'None', '2', '174', '175', '1'])
        inputs.append(['75mm', command, '1', '5', 'None', '2', '19', '20', '1'])
        inputs.append(['85mm', command, '1', '5', 'None', '1', '123', '124', '1'])
        inputs.append(['85mm', command, '1', '5', 'None', '1', '153', '154', '1'])
        inputs.append(['75mm', command, '1', '10', 'None', '1', '135', '136', '1'])
        inputs.append(['85mm', command, '1', '20', 'None', '1', '126', '127', '1'])
        inputs.append(['85mm', command, '1', '40', 'None', '1', '115', '116', '1'])
        inputs.append(['85mm', command, '1', '80', 'None', '1', '180', '181', '1'])
        inputs.append(['85mm', command, '1', '640', 'None', '1', '218', '219', '1'])
        inputs.append(['75mm', command, '1', '1280', 'None', '1', '195', '196', '1'])
        inputs.append(['85mm', command, '1', '1280', 'None', '1', '162', '163', '1'])
        inputs.append(['85mm', command, '5', '320', 'None', '1', '219', '220', '1'])
        inputs.append(['75mm', command, '5', '80', 'None', '1', '54', '55', '1'])
        inputs.append(['85mm', command, '5', '80', 'None', '1', '47', '48', '1'])
        inputs.append(['75mm', command, '5', '40', 'None', '1', '62', '63', '1'])
        inputs.append(['85mm', command, '5', '40', 'None', '1', '206', '207', '1'])
        inputs.append(['85mm', command, '5', '20', 'None', '1', '145', '146', '1'])
        inputs.append(['85mm', command, '5', '10', 'None', '1', '76', '77', '1'])




    global initial_ros_port
    global current_ros_port
    global max_ros_port

    command_list_file = command_list_file + "_temp"
    for i in range(0,len(inputs)):
        f = open(command_list_file, 'w')
        f.write(' '.join(inputs[i]) + "\n")
        f.close()
        starting_screen_counter = generate_commands_file(command_file + '_' + repr(i), problem_type, work_folder_dir,  starting_screen_counter, source_tensorflow, separate_ros_vrep_port, command_list_file)
        if (i==0):
            run_command_on_node('cat ' + command_file + '_' + repr(i) + ' > ' + command_file)
        else:
            run_command_on_node('cat ' + command_file + '_' + repr(i) + ' >> ' + command_file)
        if separate_ros_vrep_port:
            current_ros_port = current_ros_port + 1
            if current_ros_port > max_ros_port:
                current_ros_port = initial_ros_port

def main():
    opts, args = getopt.getopt(sys.argv[1:],"hefrgkv:t:d:p:s:",["dir="])
    work_folder_dir = None
    command_file = None
    execute_command_file = False
    genarate_command_file = False
    remove_stopped_process = False
    separate_ros_vrep_port = False
    source_tensorflow = False
    starting_screen_counter = 1
    force_counter = False
    k_roscore = False
    global generic_scene
    problem_type = None
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'experiment_v3.py -e |-g | -r -v <0 for specific scene 1 for generic scene> -t  -s starting_screen_counter -p problem_type -d work_folder_dir command_file'
         sys.exit()
      elif opt == '-e':
         execute_command_file = True
      elif opt == '-f' :
         force_counter = True
      elif opt == '-g':
         genarate_command_file = True
      elif opt == '-r':
          remove_stopped_process = True
      elif opt == '-k':
          k_roscore = True
      elif opt == '-v':
         separate_ros_vrep_port = True
         if(int(arg) == 1):
             generic_scene = True
      elif opt == '-t':
         source_tensorflow = arg
      elif opt == '-p':
          problem_type = arg
      elif opt == '-s':
          starting_screen_counter = int(arg)
      elif opt in ("-d", "--dir"):
         work_folder_dir = arg

    print problem_type
    command_list_file = None
    if len(args) > 0:
        command_file = args[0]
        command_list_file = "sample_input_files/" + os.path.basename(command_file)
        if not os.path.exists(command_list_file):
            command_list_file = None
    if len(args) > 1:
        command_list_file = args[1]
    if command_list_file is not None:
        generate_error_re_run_commands(command_file, problem_type, work_folder_dir,  starting_screen_counter, source_tensorflow, separate_ros_vrep_port, command_list_file)
        return

    if genarate_command_file:
        generate_commands_file(command_file, problem_type, work_folder_dir,  starting_screen_counter, source_tensorflow, separate_ros_vrep_port, command_list_file)

    if k_roscore:
        kill_roscore("run_txt_files/node_list.txt")

    if remove_stopped_process:
        running_nodes_file = "run_txt_files/running_nodes.txt"
        stopped_nodes_file = "run_txt_files/stopped_nodes.txt"
        check_finished_processes_standalone(running_nodes_file, stopped_nodes_file)

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
            run_command_file(command_file, "run_txt_files/node_list.txt",running_nodes_file, stopped_nodes_file , current_screen_counter_file, counter_list_file, force_counter)
            force_counter = False
            with open(current_screen_counter_file, 'r') as f:
                new_screen_counter = int(f.readline())
                if new_screen_counter == current_screen_counter:
                    break
                else:
                    current_screen_counter = new_screen_counter
                    #TODO automatically merge runnign and stopped nodes files using command
                    #awk 'NR==FNR{a[$0];next} !($0 in a)' stopped_nodes.txt running_nodes.txt
                    #ps axf | grep experiment_v2 | grep -v grep | awk '{print "kill -9 " $1}' | bash
                    #cat run_txt_files/stopped_nodes.txt | cut -d' ' -f2 | cut -d'_' -f1 | sort -n | awk '$1!=p+1{print p+1"-"$1-1}{p=$1}'
        while(not all_processes_stopped()):
            print "Sleeping before checking process status..."
            run_command_on_node('sleep 600')
            check_finished_processes(stopped_nodes_file)


if __name__ == '__main__':
    main()

#SSh over 2 hops
#http://www.larkinweb.co.uk/computing/mounting_file_systems_over_two_ssh_hops.html
# ssh -f userB@systemB -L 2222:systemC:22 -N
# ssh -f a0117042@sunfire-r.comp.nus.edu.sg -L 2222:unicorn0.d2.comp.nus.edu.sg:22 -N
# Dont use -f for ssh
# sshfs -p 2222 userC@localhost:/remote/path/ /mnt/localpath/
# sshfs -p 2222 neha@localhost:/data/neha/WORK_FOLDER unicorn_dir_mount/

#Periodically clean roscore
#until false; do rosclean purge -y; sleep 600; done;
#until false; do python delete_old_files.py -r -d ~/.ros/log -t 0.05; sleep 3600; done;
#python delete_old_files.py -r -d ~/WORK_FOLDER/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/ -t 0.25

#check process memory
#pmap -x pid

#To check stuck processes dur to roscore
#tail -2 ~/1[0-9][0-9][0-9]_despot_* | grep -B 2 communicate | grep despot | cut -d'/' -f4 | cut -d' ' -f1 | awk '{print "grep "$1" running_nodes.txt"}' | bash
#tail -n 2 ~/*_despot_* | grep -B 2 communicate | grep despot | cut -d'/' -f4 | cut -d' ' -f1 | awk '{print "grep "$1" running_nodes.txt"}' | bash | sed -e 's/despot_without_display/roscore/g' | awk '{print "echo ssh "$1";grep "$2" main_command_file.txt | tail -n 1"}' | bash | paste -d'|' - - | awk -F'|' '{print $1" \"" $2" \""}'

#Commands for getting joint angles of successfule pick
#find -name '*closeAndPushAction.txt'| xargs -i grep -H '*20' {}
# cat successfulPicks.txt | cut -d' ' -f15,17,1 | sed -e 's/|/: /g' | cut -d':' -f3,1 | awk '{print $2" "$3" "$1}' | sort -n | more
#cat successfulPicks.txt | cut -d' ' -f15,17,1 | sed -e 's/|/: /g' | cut -d':' -f3,1 | awk '{print $2" "$3" "$1}' | sort -n | awk '$1>1.02 && $2>1.02{print $3" "$1" "$2}' |  sort | cut -d'/' -f2 | uniq -c

#Move data
#rsync -avz ngarg211@users.ncl.sg:/big/public_share/ngarg211/WORK_FOLDER/neha_github/autonomousGrasping/grasping_ros_mico/results ./WORK_FOLDER/neha_github/autonomousGrasping/grasping_ros_mico/ > rsync.log
#Use -e 'ssh -p 2222' to specify port while rsync

#ncl node script
#tb-set-node-startcmd $n0 "/users/ngarg211/WORK_FOLDER/mystart.sh >& /tmp/mystart.log"

#awk script for sum
#awk '{s+=$1} END {print s}' mydatafile

#compress pdf file
#gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dNOPAUSE -dQUIET -dBATCH -sOutputFile=CoRlDraft6ByNeha-compressed.pdf CoRlDraft6ByNeha.pdf
