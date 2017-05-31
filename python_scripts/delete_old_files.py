import os
import sys
import datetime
import getopt
     
def main():
    delete_files = False
    directory_name = None
    opts, args = getopt.getopt(sys.argv[1:],"rd:",["dir="])
    for opt, arg in opts:
        if opt == '-r':
            delete_files = True
        if opt in ("-d", "--dir"):
            directory_name = arg
    if directory_name is None:
        print "Please specify directory name"
        return
    
    days_5 = datetime.timedelta(days = 5)
    for root, dir, files in os.walk(directory_name):
        #print "Root " + root
        for file in files:
            filename = os.path.join(root, file)
            s = os.stat(filename)
            a = datetime.datetime.now() - datetime.datetime.fromtimestamp(s.st_mtime)
            if a.total_seconds() > days_5.total_seconds():
                command = 'rm ' + filename
                print command
                if delete_files:
                    print "Executing "
                    os.system(command)

if __name__ == '__main__':
    main()