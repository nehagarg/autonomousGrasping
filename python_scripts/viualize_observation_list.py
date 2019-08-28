import os
import sys
import getopt
import argparse
import perception as perception
import matplotlib.pyplot as plt
from gqcnn import Visualizer as vis
import math
import subprocess

def visualize_observation_list(observation_list_file_name):
    if('.log' in observation_list_file_name):
        system_command = "grep 'Observation =' " + observation_list_file_name + " | cut -d'|' -f2"
        content = subprocess.check_output(["bash", "-O", "extglob", "-c", system_command])
        system_command = "grep 'Action =' " + observation_list_file_name + " | cut -d' ' -f6,7,8,9"
        action_content = subprocess.check_output(["bash", "-O", "extglob", "-c", system_command])
        content = content.strip()
        action_content = action_content.strip()
        content = content.split('\n')
        action_content = action_content.split('\n')
    else:
        with open(observation_list_file_name) as f:
            content = f.readlines()
        actions_file_name = observation_list_file_name.replace('observations', 'actions')
        action_content = []
        if(os.path.exists(actions_file_name)):
            with open(actions_file_name) as f:
                action_content = f.readlines()
                content = [x.strip() for x in content]
                action_content = [x.strip() for x in action_content]
    print content
    print action_content
    print len(content)
    print len(action_content)
    num_cols  = math.ceil(math.sqrt(len(content)))
    num_rows = math.ceil(len(content)*1.0/num_cols)
    vis.figure()
    for i in range(0,len(content)):
        ax = plt.subplot(num_rows,num_cols,i+1)
        filename_prefix = content[i]
        camera_intr =  perception.CameraIntrinsics.load(filename_prefix  + '.intr')
        color_im = perception.ColorImage.open(filename_prefix + '.npz', frame=camera_intr.frame)
        vis.imshow(color_im)
        if(i < len(action_content)):
            ax.set_title(action_content[i])

    vis.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--file_name",
                        help="Observation list file name")
    args = parser.parse_args()
    visualize_observation_list(args.file_name)

if __name__ == '__main__':
    main()
