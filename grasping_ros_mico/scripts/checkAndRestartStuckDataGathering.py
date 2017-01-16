import os
import sys
import datetime
import time

if len(sys.argv) > 1:
    filename_in = sys.argv[1]
    
    while True:
        filename = filename_in
        if filename == 'cwd':
            files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
            filename = files[-1]
            print filename
        s = os.stat(filename)
        a = datetime.datetime.now() - datetime.datetime.fromtimestamp(s.st_mtime)
        print a.total_seconds()
        if a.total_seconds() > 300:
            print "Process stuck : Setting chcktip to 1"
            os.system("rosservice call /vrep/simRosSetIntegerSignal checkTip 1")
    
        time.sleep(600)
else:
     os.system("rosservice call /vrep/simRosSetIntegerSignal checkTip 1")