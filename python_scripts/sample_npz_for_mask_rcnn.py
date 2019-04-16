import os
import sys
import getopt
import numpy as np
import perception as perception
import matplotlib.pyplot as plt
from gqcnn import Visualizer as vis
#This scipt is used to generate images that can be labelled using VIA tool for mask-RCNN

def main():
    delete_files = False
    directory_name = None
    out_directory_name = './'
    num_days = 5
    opts, args = getopt.getopt(sys.argv[1:],"d:o:",["dir=","out="])
    for opt, arg in opts:
        if opt in ("-d", "--dir"):
            directory_name = arg
        if opt in ("-o", "--out"):
            out_directory_name = arg
    if directory_name is None:
        print "Please specify directory name"
        return

    num_saved = 0
    for root, dir, files in os.walk(directory_name):
        #print "Root " + root
        for file in files:
            filename = os.path.join(root, file)
            #print filename
            if filename.endswith('.npz') and "depth" not in filename:
                rgb_image = perception.ColorImage.open(filename)
                vis.figure()
                vis.subplot(1,1,1)
                vis.imshow(rgb_image)
                vis.show(block=False)
                save_ans = raw_input("Shall I save[y|n]?")
                if save_ans == 'y':
                    png_file_name = os.path.basename(filename).replace('.npz','.png')
                    new_file_name = os.path.join(out_directory_name,png_file_name)
                    rgb_image.save(new_file_name)
                    num_saved = num_saved + 1;
                    print num_saved
                plt.close()
                #vis.imshow(rgb_image)



if __name__ == '__main__':
    main()
