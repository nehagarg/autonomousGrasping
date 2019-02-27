import os
import sys
import datetime
import getopt
import numpy as np
     
def main():
    delete_files = False
    directory_name = None
    opts, args = getopt.getopt(sys.argv[1:],"rd:t:",["dir="])
    for opt, arg in opts:
        if opt == '-r':
            delete_files = True
        if opt in ("-d", "--dir"):
            directory_name = arg
    if directory_name is None:
        print "Please specify directory name"
        return

    for root, dir, files in os.walk(directory_name):
        #print "Root " + root
        for file in files:
            filename = os.path.join(root, file)
            if filename.endswith('.npy'):
                a = np.load(filename)
                filename_new = filename
                filename_new[-1] = 'z'
                print filename_new
                command = 'rm ' + filename
                print command
                if delete_files:
                    print "Saving npz"
                    np.savez_compressed(filename_new, a)
                    print "Executing "
                    os.system(command)

if __name__ == '__main__':
    main()