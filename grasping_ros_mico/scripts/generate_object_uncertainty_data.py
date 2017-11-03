import os
import sys
import time 
def save_ans(file_name ="transparent_green_glass_uncertainty.data" ):
    with open(file_name, 'a') as f:
            with open('temp_centroid_file', 'r') as f1:
                f.write(f1.read())
                f.write('\n')
                
def interactive_query(q = 'locate left green cup', file_name ="transparent_green_glass_uncertainty.data"):                
    query = q
    while True:


        os.system('rostopic pub -1 /speech_command std_msgs/String "' + query + '"')
        a = raw_input("Save data?")
        if a == 'y':
            save_ans(file_name)
        if a == 'q':
            break
        if 'asky' in a:
            save_ans()

        if 'ask' in a:    
            query = ' '.join(a.split(' ')[1:])

def automatic_query(q = 'locate left green cup', file_name ="transparent_green_glass_uncertainty.data"):
    query = q
    i = 0
    while i < 150:
        os.system('rostopic pub -1 /speech_command std_msgs/String "' + query + '"')
        
        print('Sleeping for 10 sec before query ' + repr(i))
        
        time.sleep(10)
        save_ans(file_name)
        i = i+1

q = 'locate orange cup'
file_name ="orange_cup_uncertainty.data"   
#q = 'locate left green cup'
#file_name ="transparent_green_glass_uncertainty.data"
automatic_query(q , file_name)
#interactive_query(q , file_name )