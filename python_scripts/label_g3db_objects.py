import sys
import getopt
import os
import rospy
from vrep_common.srv import *

import yaml
    
from PySide.QtCore import *
from PySide.QtGui import *

import load_objects_in_vrep as ol

class MainWindow(QWidget):
    def __init__(self, object_file_name, dir_name, parent=None):
        QWidget.__init__(self, parent)
        self.setGeometry(300, 300, 1300, 700)
        self.lo = LabelObject(object_file_name, dir_name)
        
        
        
        self.loadNextObjectButton = QPushButton('Next', self)
        self.loadNextObjectButton.clicked.connect(self.handleNextButton)
        self.loadPrevObjectButton = QPushButton('Prev', self)
        self.loadPrevObjectButton.clicked.connect(self.handlePrevButton)
        self.addCsvParamButton = QPushButton('Add', self)
        self.addCsvParamButton.clicked.connect(self.handleAddButton)
        self.savePropertiesFileButton = QPushButton('Save', self)
        self.savePropertiesFileButton.clicked.connect(self.handleSaveButton)
        
        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.setMinimumSize(200,300)
        """
        self.addFormLayout = QFormLayout()
        self.fieldNameButton = QComboBox( self)
        self.fieldNameButton.addItem("Size")
        self.fieldNameText = QLineEdit(self)
        self.addFormLayout.addRow(self.fieldNameButton, self.fieldNameText)
        """
        self.hLayoutT = QHBoxLayout()
        self.hLayoutT.addWidget(self.plainTextEdit)
        #self.hLayoutT.addLayout(self.addFormLayout)
        
        
               
        self.hLayout = QHBoxLayout()
        self.hLayout.addWidget(self.loadNextObjectButton)
        self.hLayout.addWidget(self.loadPrevObjectButton)
        self.hLayout.addWidget(self.addCsvParamButton)
        self.hLayout.addWidget(self.savePropertiesFileButton)
        
        
        
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addLayout(self.hLayoutT)
        self.verticalLayout.addLayout(self.hLayout)
        self.setLayout(self.verticalLayout)
        self.loadObject()
        
    def loadObject(self):
        self.lo.load_next_object()
        self.setWindowTitle(os.path.basename(self.lo.object_file_names[self.lo.mesh_file_id]))
        prettyData =  yaml.dump(self.lo.yaml_out, default_flow_style=False)
        self.plainTextEdit.setPlainText(str(prettyData))
        if os.path.exists(self.lo.get_output_file_name()):
            self.savePropertiesFileButton.setText("Save Again")
        else:
            self.savePropertiesFileButton.setText("Save")
        #self.plainTextEdit.appendPlainText(str(prettyData))
        
    def handleAddButton(self):
        pass
    
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


class LabelObject:
    def __init__(self, object_file_name, dir_name):
        self.object_file_names = sorted(self.get_object_filenames(object_file_name))
        self.output_dir = dir_name
        self.mesh_file_id = 0
        self.yaml_out = {}
        
    def load_next_object(self):
        #load object
        self.yaml_out = {}
        self.yaml_out['mesh_name'] = self.object_file_names[self.mesh_file_id]
        self.yaml_out['signal_name'] = 'mesh_location'
        self.yaml_out['object_use_label'] = 'S'
        ol.update_object('load_object', self.yaml_out)
        
        #get object properties
        self.output_file_name = self.get_output_filename(self.object_file_names[self.mesh_file_id],self.output_dir)
        if(os.path.exists(self.output_file_name)):
            with open(self.output_file_name, 'r') as f:
                self.yaml_out = yaml.load(f)
        else:
            self.yaml_out['mesh_name'] = os.path.basename(self.object_file_names[self.mesh_file_id])
            
            
                
    def get_object_filenames(self, object_file_name):
        ans = []
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
                    ans.append[file]
        else:
            ans =[object_file_name]
        return ans
    
    def get_output_filename(self, object_file_name, dir_name):
        output_file_name = os.path.basename(object_file_name).replace('.obj', '.yaml')
        return os.path.join(dir_name, output_file_name)
    
    def get_output_file_name(self):
        return self.get_output_filename(self.object_file_names[self.mesh_file_id], self.output_dir)

    def save_properties_file(self, output_str):
        try:
            from yaml import CDumper as Dumper
        except ImportError:
            from yaml import Dump
        if(output_str is not None):
            self.yaml_out = yaml.load(output_str)
        output = yaml.dump(self.yaml_out, Dumper = Dumper)
        with open(self.output_file_name, 'w') as f:
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

def main():

    dir_name = './'
    opts, args = getopt.getopt(sys.argv[1:],"ho:",["outdir=",])
    app = QApplication([])
    openWindow = True
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'label_g3db_objects.py -o output_dir_name <object_file_name>'
         sys.exit()
      elif opt in ("-o", "--outdir"):
         dir_name = arg
    object_file_name = args[0]
    
    currentState = MainWindow(object_file_name, dir_name)
    currentState.show()
    app.exec_()
    
    #label_object(object_file_name, dir_name, False)
    




    #circles2 = DrawCircles()
    
    #circles2.show()
    #app.exec_()

if __name__ == '__main__':
    main()   
    
    
#Commands
# python label_g3db_objects.py -o ../grasping_ros_mico/g3db_object_labels/ ../../../vrep/G3DB_object_dataset/obj_files/