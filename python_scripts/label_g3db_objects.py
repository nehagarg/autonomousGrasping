import sys
import getopt
import os
from vrep_common.srv import *
import random

import yaml
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dump 
    
from PySide.QtCore import *
from PySide.QtGui import *
import numpy as np
import random
import load_objects_in_vrep as ol
from grasping_object_list import get_grasping_object_name_list

class MainWindow(QWidget):
    def __init__(self, object_file_name, dir_name, collect_objects, parent=None):
        QWidget.__init__(self, parent)
        self.setGeometry(300, 300, 1300, 700)
        self.lo = LabelObject(object_file_name, dir_name, collect_objects)
        
        
        
        self.loadNextObjectButton = QPushButton('Next', self)
        self.loadNextObjectButton.clicked.connect(self.handleNextButton)
        self.loadPrevObjectButton = QPushButton('Prev', self)
        self.loadPrevObjectButton.clicked.connect(self.handlePrevButton)
        self.checkCollisionButton = QPushButton('CheckCollision', self)
        self.checkCollisionButton.clicked.connect(self.handleCheckCollisionButton)
        self.checkStabilityButton = QPushButton('CheckStability', self)
        self.checkStabilityButton.clicked.connect(self.handleCheckStabilityButton)
        self.getPickPointButton = QPushButton('GetPickPoint', self)
        self.getPickPointButton.clicked.connect(self.handleGetPickPointButton)
        self.savePropertiesFileButton = QPushButton('Save', self)
        self.savePropertiesFileButton.clicked.connect(self.handleSaveButton)
        
        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.setMinimumSize(200,300)
        
        self.addFormLayout = QFormLayout()
        self.fieldNameButton = QComboBox( self)
        self.fieldNameButton.currentIndexChanged.connect(self.handleInstanceChange)
        #self.fieldNameButton.addItem("Size")
        self.fieldNameText = QPlainTextEdit(self)
        self.addFormLayout.addRow(self.fieldNameButton, self.fieldNameText)
        self.objectStableButton = QPushButton('ObjectStable', self)
        self.objectPickable = QPushButton('ObjectPickable', self)
        self.objectColliding = QPushButton('ObjectColliding', self)
        self.objectVisible = QPushButton('ObjectVisible', self)
        
        
        self.verticalLayoutTR = QVBoxLayout()
        self.verticalLayoutTR.addLayout(self.addFormLayout)
        self.verticalLayoutTR.addWidget(self.objectStableButton)
        self.verticalLayoutTR.addWidget(self.objectPickable)
        self.verticalLayoutTR.addWidget(self.objectColliding)
        self.verticalLayoutTR.addWidget(self.objectVisible)
        
        
        self.hLayoutT = QHBoxLayout()
        self.hLayoutT.addWidget(self.plainTextEdit)
        self.hLayoutT.addLayout(self.verticalLayoutTR)
        
        
               
        self.hLayout = QHBoxLayout()
        self.hLayout.addWidget(self.loadNextObjectButton)
        self.hLayout.addWidget(self.loadPrevObjectButton)
        self.hLayout.addWidget(self.savePropertiesFileButton)
        self.hLayout.addWidget(self.checkCollisionButton)
        self.hLayout.addWidget(self.checkStabilityButton)
        self.hLayout.addWidget(self.getPickPointButton)
        
        
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addLayout(self.hLayoutT)
        self.verticalLayout.addLayout(self.hLayout)
        self.setLayout(self.verticalLayout)
        self.loadObject()

    def loadObject(self, instance = None):
        self.lo.load_next_object(instance, True,True)
        self.setWindowTitle(os.path.basename(self.lo.object_file_names[self.lo.mesh_file_id]))
        prettyData =  yaml.dump(self.lo.yaml_out, default_flow_style=False)
        self.plainTextEdit.setPlainText(str(prettyData))
        if os.path.exists(self.lo.get_output_file_name()):
            self.savePropertiesFileButton.setText("Save Again")
        else:
            self.savePropertiesFileButton.setText("Save")
            
        if(instance is None):
            self.fieldNameButton.clear()
            num_instances = self.lo.get_num_instances()
            for i in range(0,num_instances):
                self.fieldNameButton.addItem(repr(i))

            self.fieldNameButton.setCurrentIndex(0)
        
        
        
        """
        #Only for debugging
        while os.path.exists(self.lo.get_output_file_name()):
            self.handleNextButton()
            self.savePropertiesFileButton.setText("Save Again")
            if(self.lo.mesh_file_id < len(self.lo.object_file_names) - 1):
                break
        """
        
        #self.plainTextEdit.appendPlainText(str(prettyData))
    
    def handleInstanceChange(self):
        text = self.fieldNameButton.currentIndex()
        print text
        self.loadObject(text)
        prettyData =  yaml.dump(self.lo.instance_yaml, default_flow_style=False)
        self.fieldNameText.setPlainText(str(prettyData))
        if('object_stable' in self.lo.instance_yaml.keys()):
            if(self.lo.instance_yaml['object_stable']):
                self.objectStableButton.setStyleSheet("background-color: green")
            else:
                self.objectStableButton.setStyleSheet("background-color: red")
        else:
            self.objectStableButton.setStyleSheet("background-color: yellow")
        if('object_pickable' in self.lo.instance_yaml.keys()):
            if(self.lo.instance_yaml['object_pickable']):
                self.objectPickable.setStyleSheet("background-color: green")
            else:
                self.objectPickable.setStyleSheet("background-color: red")
        else:
            self.objectPickable.setStyleSheet("background-color: yellow")
            
        if('colliding_with_gripper' in self.lo.instance_yaml.keys()):
            if(self.lo.instance_yaml['colliding_with_gripper']):
                self.objectColliding.setStyleSheet("background-color: red")
            else:
                self.objectColliding.setStyleSheet("background-color: green")
        else:
            self.objectColliding.setStyleSheet("background-color: yellow")
        if('object_visible' in self.lo.instance_yaml.keys()):
            if(self.lo.instance_yaml['object_visible']):
                self.objectVisible.setStyleSheet("background-color: green")
            else:
                self.objectVisible.setStyleSheet("background-color: red")
        else:
            self.objectVisible.setStyleSheet("background-color: yellow")
        
    def handleGetPickPointButton(self):
        self.lo.get_pick_point()
        prettyData =  yaml.dump(self.lo.instance_yaml, default_flow_style=False)
        self.fieldNameText.setPlainText(str(prettyData))
        
    def handleCheckCollisionButton(self):
        self.lo.check_collision()
        prettyData =  yaml.dump(self.lo.instance_yaml, default_flow_style=False)
        self.fieldNameText.setPlainText(str(prettyData))
    
    def handleCheckStabilityButton(self):
        self.lo.check_stability()
        prettyData =  yaml.dump(self.lo.instance_yaml, default_flow_style=False)
        self.fieldNameText.setPlainText(str(prettyData))
    
    def handleNextButton(self):
        if(self.lo.mesh_file_id < len(self.lo.object_file_names) - 1):
            self.lo.mesh_file_id = self.lo.mesh_file_id + 1
            self.loadObject()
            self.loadPrevObjectButton.setEnabled(True)
        else:
            self.loadNextObjectButton.setEnabled(False)
            
    def handlePrevButton(self):
        if(self.lo.mesh_file_id > 0):
            self.lo.mesh_file_id = self.lo.mesh_file_id - 1
            self.loadObject()
            self.loadNextObjectButton.setEnabled(True)
        else:
            self.loadPrevObjectButton.setEnabled(False)
    
    def handleSaveButton(self):
        try:
            self.lo.save_properties_file(self.plainTextEdit.toPlainText())
            self.savePropertiesFileButton.setText("Save Again")
        except:
            self.savePropertiesFileButton.setText("Not Saved")

#object file name gives mesh location
#dir_name gives config file location
class LabelObject:
    def __init__(self, object_file_name, dir_name, collect_objects = False):
        self.collect_objects = collect_objects
        self.object_file_names = sorted(self.get_object_filenames(object_file_name))
        self.output_dir = dir_name
        self.object_instance_dir = "object_instances"
        self.updated_object_instance_dir = "object_instances_updated"
        self.point_cloud_dir = '../grasping_ros_mico/point_clouds'
        
        self.mesh_file_id = 0
        self.yaml_out = {}
        self.size_clusters = {}
        self.size_lists = {}
        if not self.pure_shape:
            self.duplicate_file_name = os.path.dirname(object_file_name) + "/duplicate_list.yaml"
            if not os.path.exists(self.duplicate_file_name):
                self.detect_duplicates_for_object_class()
            else:
                with open(self.duplicate_file_name, 'r') as f:
                    (self.size_lists, self.size_clusters) = yaml.load(f)
                
    def load_next_object(self, instance = 0, detect_duplicates = True, load_updated_instance=False):
        if self.pure_shape:
            self.yaml_out = ol.add_object_in_scene(self.object_file_names[self.mesh_file_id], self.output_dir)
            #get object properties
            self.output_file_name = self.get_output_filename(self.object_file_names[self.mesh_file_id],self.output_dir)
            self.instance_yaml = self.yaml_out
        else:
            #load object
            self.yaml_out = {}
            self.yaml_out['mesh_name'] =  os.path.basename(self.object_file_names[self.mesh_file_id])
            self.yaml_out['mesh_dir'] = os.path.dirname(self.object_file_names[self.mesh_file_id])
            self.yaml_out['signal_name'] = 'mesh_location'
            self.yaml_out['object_use_label'] = 'S'
            ol.update_object('load_object', self.yaml_out)
            self.load_object_properties(detect_duplicates)
            self.instance_yaml = {"-" : "-"}
            if(instance is not None):
                if(load_updated_instance):
                    instance_file_name = self.get_updated_instance_file_dir(self.output_file_name)
                else:
                    instance_file_name = self.get_instance_file_dir(self.output_file_name)
                instance_file_name = instance_file_name + "/" + self.get_instance_file_name(instance)
                if(os.path.exists(instance_file_name)):
                    with open(instance_file_name, 'r') as f:
                        self.instance_yaml = yaml.load(f)
                        #self.instance_yaml['mesh_name'] = self.object_file_names[self.mesh_file_id]
                        self.instance_yaml = ol.add_object_from_properties(self.instance_yaml, True)
        
        if(self.collect_objects):
            objects_to_be_collected = []
            object_to_be_collected_filename = 'class_2.txt'#self.get_updated_instance_file_dir(self.output_file_name) + "/object_instance_names_validation1_version7.txt"
            with open(object_to_be_collected_filename, 'r') as f:
                objects_to_be_collected = [x.strip('\n') for x in f.readlines()]
            ob2 = []
            object_to_be_collected_filename = self.get_updated_instance_file_dir(self.output_file_name) + "/object_instance_names_test_version7.txt"
            with open(object_to_be_collected_filename, 'r') as f:
                ob2 = [x.strip('\n') for x in f.readlines()]
                
            if('-' not in self.instance_yaml.keys()):
                instance_name = self.get_instance_file_name(instance).replace('.yaml', '')
                print instance_name
                print objects_to_be_collected
                #if(ol.object_graspable(self.instance_yaml)):
                if instance_name in objects_to_be_collected and instance_name not in ob2:
                
                    object_class = self.get_object_class_from_mesh_name(self.instance_yaml['mesh_name'])
                    random.seed(hash(object_class))
                    c1 = random.random()
                    c2 = random.random()
                    c3 = random.random()
                    ol.update_object({'set_object_color' : [c1 ,c2,c3]},{})

                    name = self.get_instance_file_name(instance).split('.')[0].replace('-', '_')
                    ol.update_object({'set_object_name' : name}, {})
                
                        
                    
    
    def get_pick_point(self):
        ol.get_object_pick_point(self.instance_yaml)
        print "####"
        
    def check_stability(self):
        ol.check_for_object_stability(self.instance_yaml)
    
    def check_collision(self):
        ol.check_for_object_collision(self.instance_yaml)
    def check_visibility(self):
        ol.check_object_point_cloud_clipping(self.instance_yaml)
    
    def load_object_properties(self, detect_duplicates = True):
        #get object properties
        self.output_file_name = self.get_output_filename(self.object_file_names[self.mesh_file_id],self.output_dir)
        
        #Check for duplicates
        if(detect_duplicates):
            object_class = self.get_object_class_from_mesh_name(os.path.basename(self.object_file_names[self.mesh_file_id]))
            print object_class
            if object_class in self.size_clusters.keys():
                for j in range(0,len(self.size_clusters[object_class])):
                    cluster = self.size_clusters[object_class][j]
                    mesh_name = os.path.basename(self.object_file_names[self.mesh_file_id])
                    if  mesh_name in cluster:
                        mesh_index = cluster.index(mesh_name)
                        self.duplicate_mesh_name = cluster[0]
                        self.duplicate_mesh_index = mesh_index
                        self.duplicate_mesh_size = self.size_lists[object_class][j]
        else:
            self.duplicate_mesh_name = os.path.basename(self.object_file_names[self.mesh_file_id])
            self.duplicate_mesh_index = 0
            
        if(os.path.exists(self.output_file_name)):
            with open(self.output_file_name, 'r') as f:
                self.yaml_out = yaml.load(f)
        else:
            self.yaml_out['mesh_name'] = os.path.basename(self.object_file_names[self.mesh_file_id])
        self.yaml_out['duplicate_mesh_name'] = self.duplicate_mesh_name
        self.yaml_out['duplicate_mesh_index'] = self.duplicate_mesh_index
        self.yaml_out['duplicate_mesh_size'] = self.duplicate_mesh_size
    
               
    def generate_object_instance_configs(self):
        
        check_keys = set()
        for i in range(0,len(self.object_file_names)):
            self.mesh_file_id = i
            self.load_object_properties()
            check_keys.update(self.yaml_out.keys())
            self.object_instance_file_dir = self.get_instance_file_dir(self.output_file_name)
            if self.yaml_out['object_use_label'] == 'S':
                if int(self.yaml_out['duplicate_mesh_index']) == 0:
                    if 'actions' not in self.yaml_out.keys():
                        self.yaml_out['actions'] = ['noop']
                    action_lists = self.yaml_out['actions']
                    if type(action_lists[0])!=list:
                        action_lists = [self.yaml_out['actions']]
                    updated_action_lists = []    
                    for action_list in action_lists:
                        for j in range(0,len(action_list)):
                            action_ = action_list[j]
                            if(type(action_) ==  dict):
                                action = action_.keys()[0]
                                action_value = action_[action]
                                print action_
                                check_keys.add(action)
                                if(type(action_value)== str and '-' in action_value):
                                    minmax = action_value.split('-')
                                    min_val = float(minmax[0])
                                    max_val = float(minmax[1])
                                    val = random.uniform(min_val, max_val)
                                    action_list[j][action] = val
                            else:
                                check_keys.add(action_)
                                
                        updated_action_lists.append(action_list)
                    for j in range(0,len(updated_action_lists)):
                        action_list = updated_action_lists[j]
                        object_instance_file_name = self.object_instance_file_dir + "/" + self.get_instance_file_name(j)
                        yaml_val = self.yaml_out
                        if(action_list[0]!='noop'):
                            yaml_val['actions'] = action_list
                        else:
                            yaml_val['actions'] = []
                        self.save_yaml(object_instance_file_name, self.yaml_out)        
                        
        print check_keys    
            
    def update_object_instance_configs(self):
        for i in range(0,len(self.object_file_names)):
            self.mesh_file_id = i
            self.load_next_object(None)
            if self.pure_shape or (self.yaml_out['object_use_label'] == 'S' and int(self.yaml_out['duplicate_mesh_index']) == 0):
                    num_instances = self.get_num_instances()
                    for j in range(0,num_instances):
                        if not self.pure_shape:
                            self.load_next_object(j)
                        self.get_pick_point()
                        self.check_stability()
                        self.check_collision()
                        updated_object_instance_file_name = self.get_updated_instance_file_dir(self.output_file_name) + "/" + self.get_instance_file_name(j)
                        self.save_yaml(updated_object_instance_file_name, self.instance_yaml)  
    def check_visibility_for_all_objects(self):
        for i in range(0,len(self.object_file_names)):
            self.mesh_file_id = i
            self.load_next_object(None)
            if self.pure_shape or (self.yaml_out['object_use_label'] == 'S' and int(self.yaml_out['duplicate_mesh_index']) == 0):
                    num_instances = self.get_num_instances()
                    for j in range(0,num_instances):
                        if not self.pure_shape:
                            self.load_next_object(j,True,True)
                        if('-' not in self.instance_yaml.keys()):
                            if(ol.object_graspable(self.instance_yaml)):
                                self.check_visibility()
                                updated_object_instance_file_name = self.get_updated_instance_file_dir(self.output_file_name) + "/" + self.get_instance_file_name(j)
                                self.save_yaml(updated_object_instance_file_name, self.instance_yaml)  
                        
    def generate_point_clouds(self, for_classifier = False):
        for i in range(0,len(self.object_file_names)):
            self.mesh_file_id = i
            self.load_next_object(None)
            if self.pure_shape or (self.yaml_out['object_use_label'] == 'S' and int(self.yaml_out['duplicate_mesh_index']) == 0):
                num_instances = self.get_num_instances()
                object_mesh_dir = os.path.dirname(self.object_file_names[self.mesh_file_id])
                point_cloud_dir = self.point_cloud_dir
                for j in range(0,num_instances):
                        object_property_dir =  self.get_updated_instance_file_dir(self.output_file_name)
                        object_id = os.path.basename(self.get_instance_file_name(j))
                        start_range = 40
                        end_range = 41
                        if for_classifier:
                            start_range = 0
                            end_range = -1
                            file_dir = point_cloud_dir + "/" + object_id.replace('.yaml', "") 
                            if os.path.exists(file_dir):
                                end_range = 81
                        for k in range(start_range, end_range):
                            ol.save_point_cloud(object_id, object_property_dir, object_mesh_dir, point_cloud_dir, k, for_classifier)
                
    
    def get_num_instances(self):
        num_instances = 1
        if 'actions' in self.yaml_out and self.yaml_out['actions']:
            if(type(self.yaml_out['actions'][0])==list):
                num_instances = len(self.yaml_out['actions'])
        return num_instances
    
    def get_point_cloud_filename(self,i):
        instance_file_name = self.get_instance_file_name(i)
        point_cloud_name = instancce_file_name.replace('.yaml','.npy')
        return point_cloud_name
        
    def get_updated_instance_file_dir(self, object_file_dir):
        if self.pure_shape:
            return os.path.dirname(object_file_dir)
        else:
            return self.get_instance_file_dir(object_file_dir) + "/" + self.updated_object_instance_dir
    
    def get_instance_file_dir(self, object_file_dir):
        if self.pure_shape:
            return os.path.dirname(object_file_dir)
        else:
            return os.path.dirname(object_file_dir) + "/" + self.object_instance_dir
    
    def get_instance_file_name(self, i):
        if self.pure_shape:
            return self.object_file_names[self.mesh_file_id] + '.yaml'
        else:
            return self.yaml_out['mesh_name'].split('.')[0] + "_instance" + repr(i) + '.yaml'
    
    def detect_duplicates_for_object_class(self):
        self.size_lists = {}
        self.size_clusters = {}
        for i in range(0,len(self.object_file_names)):
            self.mesh_file_id = i
            object_class = self.get_object_class_from_mesh_name(os.path.basename(self.object_file_names[self.mesh_file_id]))
            self.load_next_object(None, False)
            object_size = {}
            ol.update_object("get_size", object_size)
            if(object_class in self.size_lists.keys()):
                match_found = False
                for j in range(0,len(self.size_lists[object_class])):
                    size = self.size_lists[object_class][j]
                    try:
                        np.testing.assert_almost_equal(size, object_size["object_size"],4)
                        self.size_clusters[object_class][j].append(self.yaml_out['mesh_name'])
                        match_found=True
                    except AssertionError:
                        pass
                if not match_found:
                    self.size_lists[object_class].append(object_size["object_size"])
                    self.size_clusters[object_class].append([self.yaml_out['mesh_name']])
            else:
                self.size_lists[object_class] = [object_size["object_size"]]
                self.size_clusters[object_class] = [[self.yaml_out['mesh_name']]]
        
        output = yaml.dump((self.size_lists,self.size_clusters ), Dumper = Dumper)
        with open(self.duplicate_file_name, 'w') as f:
            f.write(output)                
    
    def get_object_class_from_mesh_name(self, object_file_name):
        return object_file_name.split('-')[0]
    
    def get_object_filenames(self, object_file_name):
        ans = []
        self.pure_shape = False
        if(object_file_name == 'all_cylinders'):
            self.pure_shape = True
            ans = get_grasping_object_name_list(object_file_name)
        else:
            if(os.path.isdir(object_file_name)):
                files = [os.path.join(object_file_name, f) for f in os.listdir(object_file_name) if '.obj' in f]
                for file in files:
                    ans.append(file)
            elif(not os.path.exists(object_file_name)):
                obj_dir_name = os.path.dirname(object_file_name)
                file_prefix = os.path.basename(object_file_name)
                files = [os.path.join(obj_dir_name, f) for f in os.listdir(obj_dir_name) if f.startswith(file_prefix)]
                for file in files:
                    if '.obj' in file:
                        ans.append(file)
            else:
                ans =[object_file_name]
        return ans
    
    def get_output_filename(self, object_file_name, dir_name):
        output_file_name = os.path.basename(object_file_name).replace('.obj', '.yaml')
        return os.path.join(dir_name, output_file_name)
    
    def get_output_file_name(self):
        return self.get_output_filename(self.object_file_names[self.mesh_file_id], self.output_dir)

    def save_properties_file(self, output_str):
        
        if(output_str is not None):
            self.yaml_out = yaml.load(output_str)
        self.save_yaml(self.output_file_name, self.yaml_out)
        
    def save_yaml(self, output_file_name, yaml_val):
        output = yaml.dump(yaml_val, Dumper = Dumper)
        with open(output_file_name, 'w') as f:
            f.write(output )
            
def label_object(object_file_name, dir_name, app):
    if(os.path.isdir(object_file_name)):
        files = [os.path.join(object_file_name, f) for f in os.listdir(object_file_name) if '.obj' in f]
        for file in files:
            label_object(file, dir_name, app)
    elif(not os.path.exists(object_file_name)):
        obj_dir_name = os.path.dirname(object_file_name)
        file_prefix = os.path.basename(object_file_name)
        files = [os.path.join(obj_dir_name, f) for f in os.listdir(obj_dir_name) if f.startswith(file_prefix)]
        for file in files:
            if '.obj' in file:
                label_object(file, dir_name, app)
    else:
        print "Labeling " + os.path.basename(object_file_name)
        output_file_name = os.path.basename(object_file_name).replace('.obj', '.label')     
        if app:
            currentState = MainWindow(object_file_name)
            currentState.show()
            app.exec_()
            
        output_file_name = os.path.basename(object_file_name).replace('.obj', '.label')     
        if (os.path.exists(os.path.join(dir_name, output_file_name))):
           ans =  raw_input("Label exists. Relabel?")
           if ans=='n':
               return
        load_object(object_file_name)
        label = 'ST'
        label_in = raw_input("Please provide label: S|T|ST:")
        if label_in:
            label = label_in

        with open(os.path.join(dir_name, output_file_name), 'w') as f:
            f.write(os.path.basename(object_file_name) + ' ' + label )
        remove_object()

#object file name gives mesh location
#dir_name gives config file location
def generate_object_instance_configs(object_file_name, dir_name):
    lo = LabelObject(object_file_name, dir_name)
    lo.generate_object_instance_configs()

#object file name gives mesh location
#dir_name gives config file location
def update_object_instance_configs(object_file_name, dir_name):
    lo = LabelObject(object_file_name, dir_name)
    lo.update_object_instance_configs()

#object file name gives mesh location
#dir_name gives config file location
def generate_point_clouds(object_file_name, dir_name):
    lo = LabelObject(object_file_name, dir_name)
    lo.generate_point_clouds()

#object file name gives mesh location
#dir_name gives config file location
def generate_point_clouds_for_classification(object_file_name, dir_name):
    lo = LabelObject(object_file_name, dir_name)
    lo.point_cloud_dir = '../grasping_ros_mico/point_clouds_for_classification'
    lo.generate_point_clouds(True)

def  check_visibility_for_objects(object_file_name, dir_name):
    lo = LabelObject(object_file_name, dir_name)
    lo.check_visibility_for_all_objects()
    
#dir_name gives config file location of updated instance configs    
def generate_pickable_object_list(dir_name, version_name = ""):
    pickable_list = []
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if '.yaml' in f]
    for i in range(0,len(files)):
        object_id = os.path.basename(files[i]).replace('.yaml', '')
        object_property_dir = dir_name
        mesh_properties = ol.get_object_properties(object_id, object_property_dir)
        if(ol.object_graspable(mesh_properties, version_name)):
            pickable_list.append(object_id)
    if version_name:
        version_name = "_" + version_name
    outfile_name = dir_name + "/object_instance_names" + version_name+ ".txt"
    print outfile_name
    with open(outfile_name, 'w') as f:
        f.write("\n".join(sorted(pickable_list)))  
        f.write("\n")

def main():

    dir_name = './'
    opts, args = getopt.getopt(sys.argv[1:],"hlpqvguco:",["outdir=",])
    
    generate_object_instances = False
    update_object_instances = False
    genetate_point_clouds = False
    genetate_point_clouds_for_classification = False
    genetate_object_list = False
    collect_Selected_objects = False
    check_visibility = False
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'label_g3db_objects.py -o output_dir_name <object_file_name>'
         sys.exit()
      elif opt in ("-o", "--outdir"):
         dir_name = arg
      elif opt == '-g':
          generate_object_instances = True
      elif opt == '-u':
          update_object_instances = True
      elif opt == '-p':
          genetate_point_clouds = True
      elif opt == '-q':
          genetate_point_clouds_for_classification = True
      elif opt == '-l':
          genetate_object_list = True
      elif opt == '-c':
          collect_Selected_objects = True
      elif opt == '-v':
          check_visibility = True
    
    object_file_name = ""
    if(len(args) > 0):
        object_file_name = args[0]
    
    if(generate_object_instances):
        generate_object_instance_configs(object_file_name, dir_name)
    elif(update_object_instances):
        update_object_instance_configs(object_file_name, dir_name)
    elif(genetate_point_clouds ):
        generate_point_clouds(object_file_name, dir_name)
    elif(genetate_point_clouds_for_classification):
        generate_point_clouds_for_classification(object_file_name, dir_name)
    elif(genetate_object_list) :
        generate_pickable_object_list(dir_name, object_file_name) #object_file_name is version name
    elif(check_visibility):
        check_visibility_for_objects(object_file_name, dir_name)
    else:
        app = QApplication([])
        currentState = MainWindow(object_file_name, dir_name, collect_Selected_objects)
        currentState.show()
        app.exec_()
    
    #label_object(object_file_name, dir_name, False)
    




    #circles2 = DrawCircles()
    
    #circles2.show()
    #app.exec_()

if __name__ == '__main__':
    main()   
    
    
#Commands
# python label_g3db_objects.py -g/-u/-p/-q/-v -o ../grasping_ros_mico/g3db_object_labels/ ../../../vrep/G3DB_object_dataset/obj_files/
#python label_g3db_objects.py -l -o ../grasping_ros_mico/g3db_object_labels/object_instances/object_instances_updated/ version7
#python label_g3db_objects.py -g/-u/-p/-l/-q/-v -o ../grasping_ros_mico/pure_shape_labels all_cylinders
##python label_g3db_objects.py -l -o ../grasping_ros_mico/pure_shape_labels version7
