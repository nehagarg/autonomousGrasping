import sys
import os

def main():
    package_name = sys.argv[1]
    main_command = 'sudo apt-get download $(apt-rdepends ' + package_name+ ' | grep -v "^ " '
    os.system(main_command + ') > ' + package_name + '.log 2>&1' )
    additional_command = ""
    with open(package_name + '.log', 'r') as f:
        for line in f:
            ignore_name = line.split(' ')[7]
            additional_command = additional_command + '| grep -v "' + ignore_name + '" '
            
    os.system(main_command + additional_command +  ')')
    
if __name__ == '__main__':
    main()
    
    
"""

Url explaining how to add local directory to apt repository
http://askubuntu.com/questions/458748/is-it-possible-to-add-a-location-folder-on-my-hard-disk-to-sources-list
"""