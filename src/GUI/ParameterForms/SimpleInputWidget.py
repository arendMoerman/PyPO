from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
import sys
from numpy import array


from src.GUI.ParameterForms.InputDiscription import *
# from InputDiscription import *

Validator_floats = QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*"))
Validator_ints = QRegExpValidator(QRegExp("[-+]?[0-9]*"))


class SimpleInput(QWidget):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDiscription = inp

        self.layout = QHBoxLayout()
        self.setupUI()
        # self.makeTestBtn()
        self.setLayout(self.layout)

    def setupUI(self):
        inp = self.inputDiscription
        
        self.inputs = [QLineEdit() for k in range(inp.numFields)]
        editLayout = QHBoxLayout()
        
        for i in range(inp.numFields):
            edit = self.inputs[i]
            edit.setPlaceholderText(str(inp.hints[i]))
            editLayout.addWidget(edit)
        self.editsWid = QWidget()
        self.editsWid.setLayout(editLayout)

        self.label = self.makeLabelFromString(self.inputDiscription.label)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.editsWid)
    
    def read(self):
        l =[i.text() for i in self.inputs] 
        if self.inputDiscription.oArray:
            l = array(l)
        l = {self.inputDiscription.outputName:l}
        # print (l)
        return l

    # def makeTestBtn(self):
    #     btn = QPushButton("ok")
    #     btn.clicked.connect(self.read)
    #     self.layout.addWidget(btn)

    @staticmethod
    def makeLabelFromString(s):
        if type(s) == str:
            return QLabel(s.replace("_"," ").capitalize())
        else: return QLabel(s)
# app = QApplication(sys.argv)
# win = SimpleInput(InputDescription(inType.floats, "name", hints=["1","1","1"], numFields=3, oArray=True))

# win.show()
# app.exec_()