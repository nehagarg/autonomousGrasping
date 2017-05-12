
from PySide.QtCore import *
from PySide.QtGui import *
from log_file_parser import ParseLogFile 
import pprint
import sys
import time
import threading


class MainWindow(QWidget):
    def __init__(self, successfulTestCases, failureTestCases,parent=None):
        QWidget.__init__(self, parent)
         # setGeometry(x_pos, y_pos, width, height)
        self.setGeometry(300, 300, 1300, 700)
        self.setWindowTitle('Main')

        self.stateWindow = DrawState(successfulTestCases, [], self)
        self.stateWindow2 = DrawState([], failureTestCases, self)

        self.messageWindow = QWidget(self)
        self.messageWindow.setMaximumSize(200,600)
        actionLayout = QVBoxLayout(self.messageWindow)
        scroll = QScrollArea(self)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.messageWindow)

        self.successLabel = QLabel('Successful: ' +  repr(len(successfulTestCases)), self)
        self.failureLabel = QLabel('Failure: ' + repr(len(failureTestCases)), self)
        actionLayout.addWidget(self.successLabel)
        actionLayout.addWidget(self.failureLabel)
        
        
        layout = QVBoxLayout(self)

        hLayout = QHBoxLayout(self)
        hLayout.addWidget(self.messageWindow)
        hLayout.addWidget(self.stateWindow)
        hLayout.addWidget(self.stateWindow2)
        #hLayout.addWidget(self.beliefWindow)
        
        
        layout.addLayout(hLayout)
        #self.playTimer()

  
    
class DrawState(QWidget):
    def __init__(self, successfulTestCases, failureTestCases, parent=None):
        QWidget.__init__(self, parent)
        self.height = 600
        self.width = 500
        # setGeometry(x_pos, y_pos, width, height)
        self.setGeometry(300, 300, self.width, self.height)
        self.setWindowTitle('State')
        self.successfullTestCases = successfulTestCases 
        self.failureTestCases =  failureTestCases
        

    def paintEvent(self, event):
        paint = QPainter()
        paint.begin(self)



        # optional
        paint.setRenderHint(QPainter.Antialiasing)
        # make a white drawing background
        paint.setBrush(Qt.white)
        paint.drawRect(event.rect())
        
        #draw axis
        paint.setPen(Qt.blue)
        point1 = QPointF(self.width/2.0, 0)
        point2 = QPointF(self.width/2.0, self.height)
        point3 = QPointF(0, self.height/2.0)
        point4 = QPointF(self.width, self.height/2.0)        
        paint.drawLine(point1, point2)
        paint.drawLine(point3, point4)

        #draw bounding rectangle
        paint.setPen(Qt.gray)
        point1 = QPointF((self.width/2.0) - 170, (self.height/2.0) - 170)
        point2 = QPointF((self.width/2.0) - 170, (self.height/2.0) + 170)
        point3 = QPointF((self.width/2.0) + 170, (self.height/2.0) + 170)
        point4 = QPointF((self.width/2.0) + 170, (self.height/2.0) - 170)
        paint.drawLine(point1, point2)
        paint.drawLine(point2, point3)
        paint.drawLine(point3, point4)
        paint.drawLine(point4, point1)        
        paint.end()
        
        self.draw()


    def draw(self):
       
        
        redDim = QColor(255,0,0,0)
        self.drawCases(self.successfullTestCases, QColor(0,255,0,0))
        self.drawCases(self.failureTestCases, redDim)



        
    
    def drawCases(self, testCases, caseColor):
        paint = QPainter()
        paint.begin(self)
        paint.setPen(Qt.black)
        for i in range(0,len(testCases)):
            state = testCases[i].roundInfo_['state']
            currentState = state
            # draw object
            
            #paint.setBrush(Qt.red)
            paint.setBrush(caseColor)
            radx = state.o_r * 10        
            center = QPointF((state.x_o*-10) + (currentState.x_w*10) + (self.width/2.0) , (self.height/2.0) - (state.y_o * -10) - (currentState.y_w*10)  )
            paint.drawEllipse(center, radx, radx)

        paint.end()
def main():

    featureId = '5'
    logType = 'learning'
    if len(sys.argv) > 1:
        featureId = sys.argv[1]
    if len(sys.argv) > 2:
        logType = sys.argv[2]
    
    successfulTestCases = []
    failureTestCases = []

    for i in range(0,400):
        
        #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/adaboost_different_objects/grasping_v' + featureId + '/state_' + repr(i) + '.log'
        #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/adaboost_different_objects/sensor_observation_sum/state_' + repr(i) + '.log'
        #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_different_objects/version3/state_' + repr(i) + '.log'
        logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/graspingV4/results/despot_logs/t1_n20/Toy_test_belief_default_t1_n20_trial_' + repr(i) + '.log'
        
        #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/adaboost_different_objects' + '/state_' + repr(i) + '.log'
        #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_t10_obs_prob_change_particles_as_state_4objects.log'
        #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(i) + '_t20_obs_prob_change_particles_as_state_4objects.log'
        logParser = ParseLogFile(logfileName,'')
        if(len(logParser.stepInfo_) < 90):
            successfulTestCases.append(logParser)
        else:
            failureTestCases.append(logParser)


    app = QApplication([])
    currentState = MainWindow(successfulTestCases,failureTestCases)




    #circles2 = DrawCircles()
    currentState.show()
    #circles2.show()
    app.exec_()

if __name__ == "__main__":
    main()

