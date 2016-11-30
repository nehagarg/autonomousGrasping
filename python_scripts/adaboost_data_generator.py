
import re
import pprint
import sys
import operator
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def get_label_string():
     label_string = "-Action=ActionisDECREASEYby1, -Action=ActionisINCREASEYby1, -Action=ActionisDECREASEYby16, -Action=ActionisDECREASEXby16, -Action=ActionisINCREASEXby16, -Action=ActionisOPENGRIPPER, -Action=ActionisDECREASEXby1, -Action=ActionisCLOSEGRIPPER, -Action=ActionisINCREASEYby16, -Action=ActionisINCREASEXby1, "
     return label_string

class GripperState:
        def __init__(self, x_o_ , y_o_ , x_w_, y_w_, gripper_l_, gripper_r_, object_radius_):
            self.x_o = float(x_o_)
            self.y_o = float(y_o_)
            self.x_w = float(x_w_)
            self.y_w = float(y_w_)
            self.g_l = float(gripper_l_)
            self.g_r = float(gripper_r_)
            self.o_r = float(object_radius_)
        def __repr__(self):
            return self.__str__()

        def __str__(self):
            return "World (" + repr(self.x_w) + "," + repr(self.y_w) + ") w.r.t object (" + repr(self.x_o) + "," + repr(self.y_o) + ") gripper l,r : " + repr(self.g_l) + "," + repr(self.g_r) + " object radius " + repr(self.o_r)

class GripperObservation:
    def __init__(self, sensor_obs, gripper_l, gripper_r, x_w, y_w, x_change, y_change, terminal_state_bit):
        self.sensor_obs = [int(x) for x in sensor_obs]
        self.gripper_l_obs = int(gripper_l)
        self.gripper_r_obs = int(gripper_r)
        self.x_w_obs = int(x_w)
        self.y_w_obs = int(y_w)
        self.x_change_obs = int(x_change) ##Multiplied by 10
        self.y_change_obs = int(y_change) ##Multiplied by 10
        self.terminal_state_obs = int(terminal_state_bit)
        
class ParseLogFile:
    numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
       (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
           |
            (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
       )
     # followed by optional exponent part if desired
     (?: [Ee] [+-]? \d+ ) ?
     """
    rx =  re.compile(numeric_const_pattern, re.VERBOSE)
    
    def __init__(self, logFileName , parsePlanningBelief=True):
        self.logFileName_ = logFileName
        self.parseLogFile(logFileName, parsePlanningBelief)
    
    def createTrajectorySentence(self, trajectorySentence, ObsHash, ActionHash):
        num_steps = len(self.stepInfo_)
        if(len(self.stepInfo_) < 90):
            for i in range(0,num_steps):
                action_string = self.stepInfo_[i]['action']
                obs_string = dump(self.stepInfo_[i]['obs'].sensor_obs, Dumper = Dumper)
                hash_action_string = hash(action_string)
                hash_obs_string = hash(obs_string)
                if hash_action_string not in ActionHash:
                    ActionHash[hash_action_string] = len(ActionHash)
                if hash_obs_string not in ObsHash:
                    ObsHash[hash_obs_string] = len(ObsHash)
                #trajectorySentence.append({})
                #trajectorySentence[i]['action'] = ActionHash[hash_action_string]
                #trajectorySentence[i]['obs'] = ObsHash[hash_obs_string]
                trajectorySentence[0] = trajectorySentence[0] + '(' + repr(ActionHash[hash_action_string]) + ',' + repr(ObsHash[hash_obs_string]) + ') '
    
    def createAdaboostDataFile(self, actionNames, fileHandle):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                action_string = ((self.stepInfo_[i]['action']).replace(' ','')).replace('\n','')
                actionNames.add(action_string)
                num_left_observations = '?'
                num_right_observations = '?'
                x_w = self.roundInfo_['state'].x_w
                y_w = self.roundInfo_['state'].y_w
                g_l = self.roundInfo_['state'].g_l
                g_r = self.roundInfo_['state'].g_r
                if i > 0 :
                   num_left_observations = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[0:11]))
                   num_right_observations = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[11:22]))
                   x_w = self.stepInfo_[i-1]['state'].x_w
                   y_w = self.stepInfo_[i-1]['state'].y_w
                   g_l = self.stepInfo_[i-1]['state'].g_l
                   g_r = self.stepInfo_[i-1]['state'].g_r
                fileHandle.write(repr(x_w) + ", " + repr(y_w) + ", " + repr(g_l) + ", " + repr(g_r) + ", " + num_left_observations + ", " + num_right_observations + ", " + action_string + ".\n" )

    def createAdaboostDataFileV2(self, actionNames, fileHandle):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                action_string = ((self.stepInfo_[i]['action']).replace(' ','')).replace('\n','')
                actionNames.add(action_string)
                left_observations = ['?']*11
                right_observations = ['?']*11
                x_w = self.roundInfo_['state'].x_w
                y_w = self.roundInfo_['state'].y_w
                g_l = self.roundInfo_['state'].g_l
                g_r = self.roundInfo_['state'].g_r
                if i > 0 :
                   for j in range(0,11):
                       left_observations[j] = str(self.stepInfo_[i-1]['obs'].sensor_obs[j])
                       right_observations[j] = str(self.stepInfo_[i-1]['obs'].sensor_obs[j+11])
                   x_w = self.stepInfo_[i-1]['state'].x_w
                   y_w = self.stepInfo_[i-1]['state'].y_w
                   g_l = self.stepInfo_[i-1]['state'].g_l
                   g_r = self.stepInfo_[i-1]['state'].g_r
                fileHandle.write(repr(x_w) + ", " + repr(y_w) + ", " + repr(g_l) + ", " + repr(g_r) + ", "+ ", ".join(left_observations) + ", " + ", ".join(right_observations) + ", " + action_string + ".\n" )

    def createAdaboostDataFileV3(self, actionNames, fileHandle):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                action_string = ((self.stepInfo_[i]['action']).replace(' ','')).replace('\n','')
                actionNames.add(action_string)
                num_left_observations_upper = '?'
                num_right_observations_upper = '?'
                num_left_observations_lower = '?'
                num_right_observations_lower = '?'
                x_w = self.roundInfo_['state'].x_w
                y_w = self.roundInfo_['state'].y_w
                g_l = self.roundInfo_['state'].g_l
                g_r = self.roundInfo_['state'].g_r
                if i > 0 :
                   num_left_observations_upper = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[0:6]))
                   num_right_observations_upper = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[11:17]))
                   num_left_observations_lower = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[5:11]))
                   num_right_observations_lower = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[16:22]))
                   x_w = self.stepInfo_[i-1]['state'].x_w
                   y_w = self.stepInfo_[i-1]['state'].y_w
                   g_l = self.stepInfo_[i-1]['state'].g_l
                   g_r = self.stepInfo_[i-1]['state'].g_r
                fileHandle.write(repr(x_w) + ", " + repr(y_w) + ", " + repr(g_l) + ", " + repr(g_r) + ", " + num_left_observations_upper + ", " + num_right_observations_upper + ", " + num_left_observations_lower + ", " + num_right_observations_lower + ", " +  action_string + ".\n" )
    
    def createAdaboostDataFileV4(self, actionNames, fileHandle):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                action_string = ((self.stepInfo_[i]['action']).replace(' ','')).replace('\n','')
                actionNames.add(action_string)
                num_left_observations = '?'
                num_right_observations = '?'
                x_w = self.roundInfo_['state'].x_w
                y_w = self.roundInfo_['state'].y_w
                g_l = self.roundInfo_['state'].g_l
                g_r = self.roundInfo_['state'].g_r
                x_w_prev = 0
                y_w_prev = 0
                action_prev='?'
                if i > 0 :
                   num_left_observations = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[0:11]))
                   num_right_observations = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[11:22]))
                   x_w = self.stepInfo_[i-1]['state'].x_w
                   y_w = self.stepInfo_[i-1]['state'].y_w
                   g_l = self.stepInfo_[i-1]['state'].g_l
                   g_r = self.stepInfo_[i-1]['state'].g_r
                   action_prev = ((self.stepInfo_[i-1]['action']).replace(' ','')).replace('\n','')
                   x_w_prev = self.roundInfo_['state'].x_w
                   y_w_prev = self.roundInfo_['state'].y_w

                if i > 1 :
                   x_w_prev = self.stepInfo_[i-2]['state'].x_w
                   y_w_prev = self.stepInfo_[i-2]['state'].y_w

                fileHandle.write(repr(x_w_prev) + ", " + repr(y_w_prev) + ", " + action_prev + ", " + repr(x_w) + ", " + repr(y_w) + ", " + repr(g_l) + ", " + repr(g_r) + ", " + num_left_observations + ", " + num_right_observations + ", " + action_string + ".\n" )

    def createAdaboostDataFileV5(self, actionNames, fileHandle):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                action_string = ((self.stepInfo_[i]['action']).replace(' ','')).replace('\n','')
                actionNames.add(action_string)
                num_left_observations = '?'
                num_right_observations = '?'
                x_w = self.roundInfo_['state'].x_w
                y_w = self.roundInfo_['state'].y_w
                g_l = self.roundInfo_['state'].g_l
                g_r = self.roundInfo_['state'].g_r
                x_w_prev = 0
                y_w_prev = 0
                action_prev='?'
                x_w_prev2 = 0
                y_w_prev2 = 0
                action_prev2 = '?'
                   
                   
                if i > 0 :
                   num_left_observations = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[0:11]))
                   num_right_observations = str(sum(self.stepInfo_[i-1]['obs'].sensor_obs[11:22]))
                   x_w = self.stepInfo_[i-1]['state'].x_w
                   y_w = self.stepInfo_[i-1]['state'].y_w
                   g_l = self.stepInfo_[i-1]['state'].g_l
                   g_r = self.stepInfo_[i-1]['state'].g_r
                   action_prev = ((self.stepInfo_[i-1]['action']).replace(' ','')).replace('\n','')
                   x_w_prev = self.roundInfo_['state'].x_w
                   y_w_prev = self.roundInfo_['state'].y_w

                if i > 1 :
                   x_w_prev = self.stepInfo_[i-2]['state'].x_w
                   y_w_prev = self.stepInfo_[i-2]['state'].y_w
                   action_prev2 = ((self.stepInfo_[i-2]['action']).replace(' ','')).replace('\n','')
                   x_w_prev2 = self.roundInfo_['state'].x_w
                   y_w_prev2 = self.roundInfo_['state'].y_w
                   
                if i > 2:
                    x_w_prev2 = self.stepInfo_[i-3]['state'].x_w
                    y_w_prev2 = self.stepInfo_[i-3]['state'].y_w 
                

                fileHandle.write(repr(x_w_prev2) + ", " + repr(y_w_prev2) + ", " + action_prev2 + ", " + repr(x_w_prev) + ", " + repr(y_w_prev) + ", " + action_prev + ", " + repr(x_w) + ", " + repr(y_w) + ", " + repr(g_l) + ", " + repr(g_r) + ", " + num_left_observations + ", " + num_right_observations + ", " + action_string + ".\n" )

    def updateActionObsData(self, actionObsData, use_hash = True):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                obs_string = ''
                action_string = ''
                for j in range(0, num_steps - i):
                    if j >= len(actionObsData):
                        actionObsData.append({})
                    obs_string = obs_string + action_string + dump(self.stepInfo_[i+j]['obs'].sensor_obs, Dumper=Dumper)
                    action_string = self.stepInfo_[i+1 +j]['action']
                    if use_hash:
                        hash_obs_string = hash(obs_string)
                        hash_action_string = hash(action_string)
                    else:
                        hash_obs_string = obs_string
                        hash_action_string = action_string
                    if hash_action_string in actionObsData[j] :
                        if hash_obs_string in actionObsData[j][hash_action_string] :
                            actionObsData[j][hash_action_string][hash_obs_string] = actionObsData[j][hash_action_string][hash_obs_string] + 1
                        else:
                            actionObsData[j][hash_action_string][hash_obs_string] = 1
                    else:
                        actionObsData[j][hash_action_string] = {hash_obs_string : 1}
                    
    def updateObsActionData(self, obsActionData, use_hash = True):
        num_steps = len(self.stepInfo_) - 1
        #num_steps = 3
        if(len(self.stepInfo_) < 90):
            for i in range(0, num_steps):
                obs_string = ''
                action_string = ''
                for j in range(0, num_steps - i):
                    if j >= len(obsActionData):
                        obsActionData.append({})
                    obs_string = obs_string + action_string + dump(self.stepInfo_[i+j]['obs'], Dumper=Dumper)
                    action_string = self.stepInfo_[i+1 +j]['action']
                    if use_hash:
                        hash_obs_string = hash(obs_string)
                        hash_action_string = hash(action_string)
                    else:
                        hash_obs_string = obs_string
                        hash_action_string = action_string
                    if hash_obs_string in obsActionData[j] :
                        if hash_action_string in obsActionData[j][hash_obs_string] :
                            obsActionData[j][hash_obs_string][hash_action_string] = obsActionData[j][hash_obs_string][hash_action_string] + 1
                        else:
                            obsActionData[j][hash_obs_string][hash_action_string] = 1
                    else:
                        obsActionData[j][hash_obs_string] = {hash_action_string : 1}
        

    def getYamlWithoutBelief(self):
        stepInfoWithoutBelief = []
        for i in range(0, len(self.stepInfo_)):
            withoutBelief = self.stepInfo_[i].copy()
            withoutBelief.pop('belief')
            stepInfoWithoutBelief.append(withoutBelief)

        fullData = {'stepInfo' : stepInfoWithoutBelief, 'roundInfo' : self.roundInfo_}
        self.yamlLogWithoutBelief = dump(fullData, Dumper=Dumper)
        return self.yamlLogWithoutBelief
        
    def getYaml(self):
        fullData = {'stepInfo' : self.stepInfo_, 'roundInfo' : self.roundInfo_}
        self.yamlLog = dump(fullData, Dumper=Dumper)
        return self.yamlLog

    def dumpYaml(self):
        if self.yamlLog is None:
            getYaml()
        outFileName = self.logFileName_.rsplit('.', 1)[0] + '.yaml'
        print outFileName
        f = open(outFileName, 'w')
        f.write(self.yamlLog)

    def parsePlanningBelief(self, line, beliefArray):
        particle = re.search('Gripper at:', line)
        if particle:
            self.stateStarted = True
            values = ParseLogFile.rx.findall(line)
            #TODO : Add belief in array
            belief = {}
            belief['weight'] = values[0]
            belief['state'] = GripperState(values[1], values[2], values[7], values[8], values[3], values[4], values[5])
            beliefArray['belief'].append(belief)
        else:
            if self.stateStarted:
                self.stateStarted = False
                self.beliefParsingMode = False
                self.stateParsingMode = True
                #print len(stepInfo[stepNo]['belief'])

    def parseLearningBelief(self,line,beliefArray):
        modeEnd = re.search('INFO-thread', line)
        if modeEnd:
            self.stateStarted = False
            self.beliefParsingMode = False
            self.stateParsingMode = True
        particle = re.search('Gripper at:', line)
        if particle:
            self.stateStarted = True
            values = ParseLogFile.rx.findall(line)
            belief = {}
            belief['state'] = GripperState(values[0], values[1], values[6], values[7], values[2], values[3], values[4])
            beliefArray['belief'].append(belief)
        else:
            if self.stateStarted:
                if re.search('for State', line):
                    values = ParseLogFile.rx.findall(line)
                    belief={}
                    belief['obs'] = GripperObservation(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7])
                    beliefArray['belief'].append(belief)
                    self.stateStarted = False
            else:
                if re.search('for State', line):
                    self.stateStarted=True


    def parseLogFile(self, logFileName=None, parsePlanningBelief=True):
        if not logFileName:
            logFileName = self.logFileName_
        print logFileName
        f = open(logFileName, 'r')
        #print f
        fileText = f.readlines()
        self.beliefParsingMode = False
        self.stateStarted = False
        self.stateParsingMode = False
        self.initialStateParsingMode = False
        stepInfo=[]
        roundInfo= {}
        stepNo = -1
        for line in fileText:
            roundStart = re.match('Initial state:', line)
            if roundStart:
                self.initialStateParsingMode = True

            stepStart = re.search('Round \d+ Step (\d+)', line)
            if stepStart:
                self.beliefParsingMode = True
                stepNo = int(stepStart.group(1))
                stepInfo.append({})
                stepInfo[stepNo]['belief'] = []
                #stepInfo[stepNo]['state'] = GripperState()  
              
            if self.initialStateParsingMode:
                if re.search('Gripper at:', line):
                    print line
                    values = ParseLogFile.rx.findall(line)
                    roundInfo['state'] = GripperState(values[0], values[1], values[6], values[7], values[2], values[3], values[4])
                    self.initialStateParsingMode = False
            
            if self.beliefParsingMode:
                if parsePlanningBelief:
                    self.parsePlanningBelief(line,stepInfo[stepNo])
                else:
                    self.parseLearningBelief(line,stepInfo[stepNo])

            if self.stateParsingMode:
                if re.match('- Action = ', line):
                    stepInfo[stepNo]['action'] = line
                if re.search('Gripper at:', line):
                    if not stepInfo[stepNo].has_key('state'):
                        values = ParseLogFile.rx.findall(line)
                        stepInfo[stepNo]['state'] = GripperState(values[0], values[1], values[6], values[7], values[2], values[3], values[4])
                if re.search('for State', line):
                    values = ParseLogFile.rx.findall(line)
                    stepInfo[stepNo]['obs'] = GripperObservation(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7])
                if re.search('- ObsProb', line):
                    values = ParseLogFile.rx.findall(line)
                    stepInfo[stepNo]['obsProb'] = float(values[0])
                if re.search('- Reward', line):
                    values = ParseLogFile.rx.findall(line)
                    stepInfo[stepNo]['reward'] = float(values[0])
                    self.stateParsingMode = False
                
                    
                
                    
                
        #print len(stepInfo)
        pprint.pprint( roundInfo)
        self.roundInfo_ = roundInfo
        self.stepInfo_ = stepInfo
                    
            #else:
                #print "Not matched " + line 

def dumpAllPlanningLogs():
    stateId = '0'
    if len(sys.argv) > 1:
        stateId = sys.argv[1]

    allLogs = []
    obsActionData = []
    actionObsData = []
    trajectorySentences = []
    ActionHash = {}
    ObsHash = {}
    f = open('grasping_v5.data', 'w')
    actionNames = set([])

    for i in range(0,400):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(i) + '_t20_obs_prob_change_particles_as_state_4objects.log' 
        lfp = ParseLogFile(logfileName)
        #trajectorySentence = ['']
        #lfp.createTrajectorySentence(trajectorySentence, ObsHash, ActionHash)
        #trajectorySentences.append(trajectorySentence)
        #lfp.updateObsActionData(obsActionData, False)
        #lfp.updateActionObsData(actionObsData, False)
        #allLogs.append(lfp.getYamlWithoutBelief())

        lfp.createAdaboostDataFileV5(actionNames, f)
    f.close()
    
    label_string = ""
    for action in actionNames:
        label_string = label_string + action + ", "
    #order of elemnts in set might not be same everytime, hence listing of action names
    label_string = "-Action=ActionisDECREASEYby1, -Action=ActionisINCREASEYby1, -Action=ActionisDECREASEYby16, -Action=ActionisDECREASEXby16, -Action=ActionisINCREASEXby16, -Action=ActionisOPENGRIPPER, -Action=ActionisDECREASEXby1, -Action=ActionisCLOSEGRIPPER, -Action=ActionisINCREASEYby16, -Action=ActionisINCREASEXby1, "

    f1 = open('grasping.names', 'w')
    f1.write(label_string[:-2] + ".\n")
    f1.write("x_w: continuous.\n")
    f1.write("y_w: continuous.\n")
    f1.write("g_l: continuous.\n")
    f1.write("g_r: continuous.\n")
    f1.write("obs_count_l: continuous.\n")
    f1.write("obs_count_r: continuous.\n")
    f1.close()   

    f1 = open('grasping_v2.names', 'w')
    f1.write(label_string[:-2] + ".\n")
    f1.write("x_w: continuous.\n")
    f1.write("y_w: continuous.\n")
    f1.write("g_l: continuous.\n")
    f1.write("g_r: continuous.\n")
    for j in range(0,22):
        f1.write("obs_sensor_" + str(j) + ": 0, 1.\n")
    f1.close()   

    f1 = open('grasping_v3.names', 'w')
    f1.write(label_string[:-2] + ".\n")
    f1.write("x_w: continuous.\n")
    f1.write("y_w: continuous.\n")
    f1.write("g_l: continuous.\n")
    f1.write("g_r: continuous.\n")
    f1.write("obs_count_l_u: continuous.\n")
    f1.write("obs_count_r_u: continuous.\n")
    f1.write("obs_count_l_l: continuous.\n")
    f1.write("obs_count_r_l: continuous.\n")
    f1.close()

    f1 = open('grasping_v4.names', 'w')
    f1.write(label_string[:-2] + ".\n")
    f1.write("x_w_prev: continuous.\n")
    f1.write("y_w_prev: continuous.\n")
    f1.write("action_prev: " + label_string[:-2] + ".\n")
    f1.write("x_w: continuous.\n")
    f1.write("y_w: continuous.\n")
    f1.write("g_l: continuous.\n")
    f1.write("g_r: continuous.\n")
    f1.write("obs_count_l: continuous.\n")
    f1.write("obs_count_r: continuous.\n")
    f1.close() 

    f1 = open('grasping_v5.names', 'w')
    f1.write(label_string[:-2] + ".\n")
    f1.write("x_w_prev2: continuous.\n")
    f1.write("y_w_prev2: continuous.\n")
    f1.write("action_prev2: " + label_string[:-2] + ".\n")
    f1.write("x_w_prev: continuous.\n")
    f1.write("y_w_prev: continuous.\n")
    f1.write("action_prev: " + label_string[:-2] + ".\n")
    f1.write("x_w: continuous.\n")
    f1.write("y_w: continuous.\n")
    f1.write("g_l: continuous.\n")
    f1.write("g_r: continuous.\n")
    f1.write("obs_count_l: continuous.\n")
    f1.write("obs_count_r: continuous.\n")
    f1.close() 
    #f = open('./obsActionAnalysis/trajectorySentencesOnlyTouchObs.txt', 'w')
    #output = dump(trajectorySentences, Dumper = Dumper)
    #f.write(output)

    #obsCountArray = []
    #obsCountArrays = []
    #num_steps = len(obsActionData)
    ##num_steps = 1
    #for i in range(0, num_steps):
    #    obsCountArrays.append([])
    #    for obs in obsActionData[i].keys(): 
    #        obsInfo = {'history_length': i, 'obs': obs, 'actions' : obsActionData[i][obs]}
    #        obsCount = 0
    #        for action in obsActionData[i][obs].keys():
    #            obsCount = obsCount+obsActionData[i][obs][action]
    #        obsInfo['obsCount'] = obsCount
    #        obsCountArray.append(obsInfo)
    #        obsCountArrays[i].append(obsInfo)

    #sorted_obsCountArray = sorted(obsCountArray,key=operator.itemgetter('obsCount'), reverse=True)
    #output = dump(sorted_obsCountArray, Dumper=Dumper)
    #f = open('obsCount.yaml', 'w')
    #f.write(output)

    #for i in range(0, len(obsCountArrays)):
    #    sorted_obsCountArray = sorted(obsCountArrays[i],key=operator.itemgetter('obsCount'), reverse=True)
    #    output = dump(sorted_obsCountArray, Dumper=Dumper)
    #    f = open('./obsActionAnalysis/obsCountLength' + repr(i) + '.yaml', 'w')
    #    f.write(output)

    #pprint.pprint(allLogs)
    #for i in range(0, len(obsActionData)):
    #    output = dump(obsActionData[i], Dumper=Dumper)
    #    f = open('./obsActionAnalysis/length_hash' + repr(i) + '.yaml', 'w')
    #    f.write(output)

    #for i in range(0, len(actionObsData)):
    #    output = dump(actionObsData[i], Dumper=Dumper)
    #    f = open('./obsActionAnalysis/ActionSensorObs_length' + repr(i) + '.yaml', 'w')
    #    f.write(output)

 
   
def readLearningLog():
    print "TOSO"
def main():
    dumpAllPlanningLogs()

if __name__=="__main__":
    main()
