
from PySide.QtCore import *
from PySide.QtGui import *
from log_file_parser import ParseLogFile 
import pprint
import sys
import time
import threading


class MainWindow(QWidget):
    def __init__(self, logfileName, drawPlanningBelief = True, parent=None):
        QWidget.__init__(self, parent)
         # setGeometry(x_pos, y_pos, width, height)
        self.setGeometry(300, 300, 1300, 700)
        self.setWindowTitle('Main')
        self.drawPlanningBelief = drawPlanningBelief

        self.nextButton = QPushButton('Next', self)
        self.nextButton.clicked.connect(self.handleNextButton)

        self.prevButton = QPushButton('Prev', self)
        self.prevButton.clicked.connect(self.handlePrevButton)

        self.playButton = QPushButton('Play', self)
        self.playButton.clicked.connect(self.handlePlayButton)
        
        self.playButtonState = 0
        self.playButtonTimers = [threading.Timer(1, self.playTimer), threading.Timer(1, self.playTimer)]
        self.currentTimer = 0
        
        belief_type = ''
        if drawPlanningBelief:
            belief_type = 'planning'
        self.logParser = ParseLogFile(logfileName,belief_type)
        self.stateWindow = DrawState(self.logParser, self)
        if self.drawPlanningBelief:
            self.beliefWindow = DrawPlanningBelief(self.logParser, self)
        else:
            self.beliefWindow = DrawLearningBelief(self.logParser, self)
        
        self.messageWindow = QWidget(self)
        self.messageWindow.setMaximumSize(200,600)
        actionLayout = QVBoxLayout(self.messageWindow)
        scroll = QScrollArea(self)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.messageWindow)

        self.currentStepLabel = QLabel('Current Step: ' + repr(self.stateWindow.step) + '/' + repr(len(self.logParser.stepInfo_)), self)
        self.currentActionLabel = QLabel('Action: ', self)
        self.nextActionLabel = QLabel('0:' + self.logParser.stepInfo_[0]['action'][20:], self)
        actionLayout.addWidget(self.currentStepLabel)
        actionLayout.addWidget(self.currentActionLabel)
        actionLayout.addWidget(self.nextActionLabel)
        
        #actionLayout.addWidget(QLabel('Actions', self))
        self.actionLabels=[]
        for i in range(0, len(self.logParser.stepInfo_)):
            actionText = repr(i) + ":" + self.logParser.stepInfo_[i]['action'][20:]
            self.actionLabels.append(QLabel(actionText, self))
            #actionLayout.addWidget(self.actionLabels[i])
    
        
        layout = QVBoxLayout(self)

        hLayout = QHBoxLayout(self)
        hLayout.addWidget(self.messageWindow)
        hLayout.addWidget(self.stateWindow)
        hLayout.addWidget(self.beliefWindow)
        
        
        hLayoutB = QHBoxLayout(self)
        hLayoutB.addWidget(self.playButton)
        hLayoutB.addWidget(self.prevButton)
        hLayoutB.addWidget(self.nextButton)
        
        
        layout.addLayout(hLayout)
        layout.addLayout(hLayoutB)
        #self.playTimer()

    def playTimer(self):
        if self.playButtonState == 0:
            return
        self.handleNextButton()
        if self.stateWindow.step >= len(self.logParser.stepInfo_) - 1:
            return
        self.currentTimer = (self.currentTimer + 1) % 2
        self.playButtonTimers[self.currentTimer] =  threading.Timer(1, self.playTimer)
        self.playButtonTimers[self.currentTimer].start()
            
        
    def handlePlayButton(self):
        
        if self.playButtonState == 0:
            self.playButton.setText('Pause')
            self.playButtonState = 1
            self.playTimer()
            
        else:
            self.playButton.setText('Play')
            self.playButtonState = 0
            self.playButtonTimers[self.currentTimer].cancel()

    def handleNextButton(self):
        self.stateWindow.incStep()
        self.stateWindow.update()
        self.beliefWindow.incStep()
        self.beliefWindow.update()
        self.prevButton.setEnabled(True)
        if self.stateWindow.step >= len(self.logParser.stepInfo_) - 1:
            self.nextButton.setDisabled(True)
        if self.stateWindow.step > -1 and self.stateWindow.step <= len(self.logParser.stepInfo_) - 1:
            actionText = repr(self.stateWindow.step) + ":" + self.logParser.stepInfo_[self.stateWindow.step]['action'][20:]
            self.currentActionLabel.setText(actionText)
        if self.stateWindow.nextStep > -1 and self.stateWindow.nextStep <= len(self.logParser.stepInfo_) - 1:    
            nextActionText = repr(self.stateWindow.nextStep) + ":" + self.logParser.stepInfo_[self.stateWindow.nextStep]['action'][20:]
            self.nextActionLabel.setText(nextActionText)
        self.currentStepLabel.setText('Current Step: ' + repr(self.stateWindow.step) + '/' + repr(len(self.logParser.stepInfo_)))
        
        

    def handlePrevButton(self):
        self.stateWindow.decStep()
        self.stateWindow.update()
        self.beliefWindow.decStep()
        self.beliefWindow.update()
        self.nextButton.setEnabled(True)
        if self.stateWindow.step <= -1:
            self.prevButton.setDisabled(True)
        if self.stateWindow.step > -1 and self.stateWindow.step <= len(self.logParser.stepInfo_) - 1:
            actionText = repr(self.stateWindow.step) + ":" + self.logParser.stepInfo_[self.stateWindow.step]['action'][20:]
            self.currentActionLabel.setText(actionText)
        if self.stateWindow.nextStep > -1 and self.stateWindow.nextStep <= len(self.logParser.stepInfo_) - 1:    
            nextActionText = repr(self.stateWindow.nextStep) + ":" + self.logParser.stepInfo_[self.stateWindow.nextStep]['action'][20:]
            self.nextActionLabel.setText(nextActionText)
        self.currentStepLabel.setText('Current Step: ' + repr(self.stateWindow.step) + '/' + repr(len(self.logParser.stepInfo_)))

    
class DrawState(QWidget):
    def __init__(self, logParser, parent=None):
        QWidget.__init__(self, parent)
        self.height = 600
        self.width = 500
        # setGeometry(x_pos, y_pos, width, height)
        self.setGeometry(300, 300, self.width, self.height)
        self.setWindowTitle('State')
        self.step = -1
        self.nextStep = 0
        self.roundInfo = logParser.roundInfo_
        self.stepInfo =  logParser.stepInfo_
        
        
        

    def incStep(self):
        self.step = self.step + 1
        self.updateNextStep()

    def decStep(self):
        self.step = self.step -1
        self.updateNextStep()

    def updateNextStep(self):
        self.nextStep = self.step + 1;
        if self.step < -1 :
            self.step = -1
            self.nextStep = 0
        if self.step >= len(self.stepInfo) - 1:
            self.step = len(self.stepInfo) - 1
            self.nextStep = self.step

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
        paint = QPainter()
        paint.begin(self)

        if self.step == -1 :
            state = self.roundInfo['state']
            
        else:
            state = self.stepInfo[self.step]['state']
            

        #draw path
        path = QPainterPath()
        paint.setPen(Qt.green)
        initialState = QPointF((self.roundInfo['state'].x_w)*10 + (self.width/2.0), (self.height/2.0) - (self.roundInfo['state'].y_w)*10)
        path.moveTo(initialState)
        for i in range(0, len(self.stepInfo)):
            path.lineTo(QPointF((self.stepInfo[i]['state'].x_w)*10 + (self.width/2.0), (self.height/2.0) - (self.stepInfo[i]['state'].y_w)*10))

        #paint.drawPath(path)

        #Mark Initial Point
        paint.drawEllipse(initialState,3 ,3)

        paint.setPen(Qt.black)
        
        

        # draw object
        paint.setBrush(Qt.red)
        
        radx = state.o_r * 10        
        center = QPointF((state.x_o*-10) + (state.x_w*10) + (self.width/2.0) , (self.height/2.0) - (state.y_o * -10) - (state.y_w*10)  )
        paint.drawEllipse(center, radx, radx)
        
        #draw gripper
        gripperPoint1 = QPointF((state.x_w - state.g_l)*10 + (self.width/2.0), (self.height/2.0) - (state.y_w + 10)*10)
        gripperPoint2 = QPointF((state.x_w - state.g_l)*10 + (self.width/2.0), (self.height/2.0) - (state.y_w)*10)
        gripperPoint3 = QPointF((state.x_w + state.g_r)*10 + (self.width/2.0), (self.height/2.0) - (state.y_w )*10)
        gripperPoint4 = QPointF((state.x_w + state.g_r)*10 + (self.width/2.0), (self.height/2.0) - (state.y_w + 10 )*10)
        
        paint.drawLine(gripperPoint1, gripperPoint2)
        paint.drawLine(gripperPoint2, gripperPoint3)
        paint.drawLine(gripperPoint3, gripperPoint4)
        
        #draw observation
        if(self.step == -1):
            obs_array = [0 for x in range(0,22)]
        else:
            obs_array = self.stepInfo[self.step]['obs'].sensor_obs
        for i in range(0,2):
            for j in range(0,11):
                if(obs_array[i*11 + j] == 0):
                    paint.setBrush(Qt.black)
                else:
                    paint.setBrush(Qt.yellow)
                if i == 0:
                    sensorPoint = QPointF((state.x_w - state.g_l)*10 + (self.width/2.0), (self.height/2.0) - (state.y_w + j)*10)
                else:
                    sensorPoint = QPointF((state.x_w + state.g_r)*10 + (self.width/2.0), (self.height/2.0) - (state.y_w + j)*10)
                paint.drawEllipse(sensorPoint,3 ,3)
        paint.end()

class DrawLearningBelief(DrawState):
    def draw(self):
        
        
        if self.step == -1 :
            currentState = self.roundInfo['state']
            
            
        else:
            currentState = self.stepInfo[self.step]['state']
            
        beliefState = self.stepInfo[self.nextStep]['belief']
        particleState = beliefState[0]['state']
        path = QPainterPath()
        
        
        paint = QPainter()
        paint.begin(self)
        paint.setPen(Qt.black)

        #drawgripper
        gripperPoint1 = QPointF((particleState.x_w - particleState.g_l)*10 + (self.width/2.0), (self.height/2.0) - (particleState.y_w + 10)*10)
        gripperPoint2 = QPointF((particleState.x_w - particleState.g_l)*10 + (self.width/2.0), (self.height/2.0) - ((particleState.y_w)*10))
        gripperPoint3 = QPointF((particleState.x_w + particleState.g_r)*10 + (self.width/2.0), (self.height/2.0) - ((particleState.y_w)*10))
        gripperPoint4 = QPointF((particleState.x_w + particleState.g_r)*10 + (self.width/2.0), (self.height/2.0) - ((particleState.y_w + 10 )*10))
        
        paint.drawLine(gripperPoint1, gripperPoint2)
        paint.drawLine(gripperPoint2, gripperPoint3)
        paint.drawLine(gripperPoint3, gripperPoint4)
        
        initialState = QPointF((particleState.x_w)*10 + (self.width/2.0), (self.height/2.0) - (particleState.y_w)*10)
        path.moveTo(initialState)

        for j in range(1,len(beliefState)):
            i = len(beliefState)-j
            obs = beliefState[i]['obs']
            #drawgripper
            gripperPoint1 = QPointF((obs.x_w_obs - obs.gripper_l_obs)*10 + (self.width/2.0), (self.height/2.0) - (obs.y_w_obs + 10)*10)
            gripperPoint2 = QPointF((obs.x_w_obs - obs.gripper_l_obs)*10 + (self.width/2.0), (self.height/2.0) - ((obs.y_w_obs)*10))
            gripperPoint3 = QPointF((obs.x_w_obs + obs.gripper_r_obs)*10 + (self.width/2.0), (self.height/2.0) - ((obs.y_w_obs)*10))
            gripperPoint4 = QPointF((obs.x_w_obs + obs.gripper_r_obs)*10 + (self.width/2.0), (self.height/2.0) - ((obs.y_w_obs + 10 )*10))
        
            paint.drawLine(gripperPoint1, gripperPoint2)
            paint.drawLine(gripperPoint2, gripperPoint3)
            paint.drawLine(gripperPoint3, gripperPoint4)
            pathPoint = QPointF((obs.x_w_obs)*10 + (self.width/2.0), (self.height/2.0) - (obs.y_w_obs)*10)
            path.lineTo(pathPoint)
        
        #drawPath
        paint.setPen(Qt.green)
        paint.drawPath(path)
        
        #draw obs
        paint.setPen(Qt.black)
        for k in range(1,len(beliefState)):
            obs = beliefState[len(beliefState)-k]['obs']
            obs_array= obs.sensor_obs
            for i in range(0,2):
                for j in range(0,11):
                    if(obs_array[i*11 + j] == 0):
                        paint.setBrush(Qt.black)
                    else:
                        paint.setBrush(Qt.yellow)
                    if i == 0:
                        sensorPoint = QPointF((obs.x_w_obs - obs.gripper_l_obs)*10 + (self.width/2.0), (self.height/2.0) - (obs.y_w_obs + j)*10)
                    else:
                        sensorPoint = QPointF((obs.x_w_obs + obs.gripper_r_obs)*10 + (self.width/2.0), (self.height/2.0) - (obs.y_w_obs + j)*10)
                    paint.drawEllipse(sensorPoint,3 ,3)

        # draw object
        paint.setBrush(Qt.red)
        
        radx = particleState.o_r * 10        
        center = QPointF((particleState.x_o*-10) + (particleState.x_w*10) + (self.width/2.0) , (self.height/2.0) - (particleState.y_o * -10) - (particleState.y_w*10)  )
        paint.drawEllipse(center, radx, radx)

        paint.end()
        
class DrawPlanningBelief(DrawState):
    def draw(self):
        paint = QPainter()
        paint.begin(self)
        paint.setPen(Qt.black)
        
        if self.step == -1 :
            currentState = self.roundInfo['state']
            
        else:
            currentState = self.stepInfo[self.step]['state']
        
        beliefState = self.stepInfo[self.nextStep]['belief']
        max_weight = 0.0
        for i in range(0,len(beliefState)):
            
            if float(beliefState[i]['weight']) > max_weight:
                max_weight = float(beliefState[i]['weight'])

            state = beliefState[i]['state']
            #draw gripper

            gripperPoint1 = QPointF((currentState.x_w - state.g_l)*10 + (self.width/2.0), (self.height/2.0) - (currentState.y_w + 10)*10)
            gripperPoint2 = QPointF((currentState.x_w - state.g_l)*10 + (self.width/2.0), (self.height/2.0) - ((currentState.y_w)*10))
            gripperPoint3 = QPointF((currentState.x_w + state.g_r)*10 + (self.width/2.0), (self.height/2.0) - ((currentState.y_w)*10))
            gripperPoint4 = QPointF((currentState.x_w + state.g_r)*10 + (self.width/2.0), (self.height/2.0) - ((currentState.y_w + 10 )*10))
        
            paint.drawLine(gripperPoint1, gripperPoint2)
            paint.drawLine(gripperPoint2, gripperPoint3)
            paint.drawLine(gripperPoint3, gripperPoint4)


        for i in range(0,len(beliefState)):
            state = beliefState[i]['state']
            # draw object
            
            #paint.setBrush(Qt.red)
            paint.setBrush(QColor(255,0,0,( float(beliefState[i]['weight'])*0.99/(max_weight+0.00001))*250))
            radx = state.o_r * 10        
            center = QPointF((state.x_o*-10) + (currentState.x_w*10) + (self.width/2.0) , (self.height/2.0) - (state.y_o * -10) - (currentState.y_w*10)  )
            paint.drawEllipse(center, radx, radx)
        paint.end()
    



def main():

    stateId = '0'
    logType = 'planning'
    if len(sys.argv) > 1:
        stateId = sys.argv[1]
    if len(sys.argv) > 2:
        logType = sys.argv[2]
    
    logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/graspingV4/results/despot_logs/t1_n20/Toy_test_belief_default_t1_n20_trial_' + stateId + '.log'
    #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/graspingV3_state_' + stateId + '_t120_obs_prob_change_particles_as_state_5objects.log' 
    #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/single_object_belief/training_objects/graspingV4_state_' + stateId + '_t10_n10_obs_prob_change_single_object_radius_1_4objects_100_particles.log' 
    #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(stateId) + '_t20_obs_prob_change_particles_as_state_4objects.log'
    if(logType == 'planning_train_t10_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + stateId + '_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType == 'planning_train_t1_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + stateId + '_t1_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType == 'planning_test_t10_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_different_objects_obs_prob_change_particles_as_state/graspingV4_state_' + stateId + '_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType == 'planning_test_t1_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_different_objects_obs_prob_change_particles_as_state/graspingV4_state_' + stateId + '_t1_n10_obs_prob_change_particles_as_state_4objects.log'
    
    if(logType == 'planning_train_t1_n5'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + stateId + '_t1_n5_obs_prob_change_particles_as_state_4objects.log'
    if(logType=='learning_train_dagger_version7_t10_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_' + stateId + '_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType=='learning_train_dagger_version7_t1_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_' + stateId + '_t1_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType=='learning_train_dagger_version7_t1_n5'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_' + stateId + '_t1_n5_obs_prob_change_particles_as_state_4objects.log'
    if(logType=='learning_test_dagger_version7_t10_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_different_objects/version7/dagger_data/graspingV4_state_' + stateId + '_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType=='learning_test_dagger_version7_t1_n10'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_different_objects/version7/dagger_data/graspingV4_state_' + stateId + '_t1_n10_obs_prob_change_particles_as_state_4objects.log'
    if(logType=='learning_test_dagger_version7_t1_n5'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_different_objects/version7/dagger_data/graspingV4_state_' + stateId + '_t1_n5_obs_prob_change_particles_as_state_4objects.log'


    
    if(logType=='learning'):
        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/5_objects_obs_prob_change_particles_as_state/learning/state_' + stateId + '.log' 
    #logfileName = '../despot-0.2/test.log'
    app = QApplication([])
    currentState = MainWindow(logfileName, logType!='learning')




    #circles2 = DrawCircles()
    currentState.show()
    #circles2.show()
    app.exec_()

if __name__ == "__main__":
    main()

