
import re
import pprint
import sys
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


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
    
    def __init__(self, logFileName , parsePlanningBelief=True, round_no = 0):
        self.logFileName_ = logFileName
        self.parseLogFile(logFileName, parsePlanningBelief, round_no)
    
    def getFullDataWithoutBelief(self):
        stepInfoWithoutBelief = []
        for i in range(0, len(self.stepInfo_)):
            withoutBelief = self.stepInfo_[i].copy()
            
            withoutBelief.pop('belief')
            stepInfoWithoutBelief.append(withoutBelief)

        fullData = {'stepInfo' : stepInfoWithoutBelief, 'roundInfo' : self.roundInfo_}
        return fullData

    def getYamlWithoutBelief(self):
        
        fullData = getFullDataWithoutBelief()
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


    def parseLogFile(self, logFileName=None, parsePlanningBelief=True, round_no = 0):
        if not logFileName:
            logFileName = self.logFileName_
        #print logFileName
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
            round_start_expression = 'Initial state:'
            #if round_no >=0:
            #    round_start_expression = ' Round ' + repr(round_no) + ' '
            #print round_start_expression
            roundStart = re.match(round_start_expression, line)
            #print roundStart
            if roundStart and round_no < 0: #round_no -1 for getting information of last round
                #print line
                self.initialStateParsingMode = True
                stepInfo=[]
                roundInfo={}
                stepNo = -1
            regular_expression = 'Round \d+ Step (\d+)'
            if round_no >=0:
                regular_expression = 'Round ' + repr(round_no) + ' Step (\d+)'
            stepStart = re.search(regular_expression, line)
            if stepStart:
                #print line
                self.beliefParsingMode = True
                stepNo = int(stepStart.group(1))
                stepInfo.append({})
                stepInfo[stepNo]['belief'] = []
                #stepInfo[stepNo]['state'] = GripperState()  
              
            if self.initialStateParsingMode:
                if re.search('Gripper at:', line):
                    #print line
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
        #pprint.pprint( roundInfo)
        self.roundInfo_ = roundInfo
        self.stepInfo_ = stepInfo
                    
            #else:
                #print "Not matched " + line 

def dumpAllPlanningLogs():
    stateId = '0'
    if len(sys.argv) > 1:
        stateId = sys.argv[1]

    allLogs = []
    #for i in range(0,500):
    #    logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(i) + '_t120_obs_prob_change_particles_as_state_5objects.log' 
    for i in range(0,1):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(i) + '_t20_obs_prob_change_particles_as_state_4objects.log'
        lfp = ParseLogFile(logfileName)        
        allLogs.append(lfp.getYamlWithoutBelief())

    #pprint.pprint(allLogs)
    output = dump(allLogs, Dumper=Dumper)
    f = open('test.yaml', 'w')
    f.write(output)

def readLearningLog():
    print "TOSO"
def main():
    dumpAllPlanningLogs()

if __name__=="__main__":
    main()
