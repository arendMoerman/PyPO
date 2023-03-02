import os
import sys
import shutil
import asyncio

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QScrollArea
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from PyQt5.QtCore import Qt

from src.GUI.ParameterForms import formGenerator
from src.GUI.ParameterForms.InputDescription import InputDescription
from src.GUI.utils import inType
import src.GUI.ParameterForms.formDataObjects as fDataObj
from src.GUI.PlotScreen import PlotScreen
from src.GUI.TransformationWidget import TransformationWidget
from src.GUI.Acccordion import Accordion
from src.GUI.ElementWidget import ReflectorWidget, FrameWidget, FieldsWidget, CurrentWidget, SFieldsWidget, SymDialog
from src.GUI.Console import ConsoleGenerator

from src.GUI.Console import print
import numpy as np
from src.PyPO.Checks import InputReflError, InputRTError

sys.path.append('../')
sys.path.append('../../')
import src.PyPO.System as st
import src.PyPO.Threadmgr as TManager




##
# @file 
# defines classes PyPOMainWindow and MainWidget
# PyPOMainWindow is responsible for setting up the window and toolbars
# MainWidget is responsble for all gui functionalities
#
class MainWidget(QWidget):
    ##
    # Constructot. Configures the layout and initialises the underlying system
    # @see System
    # 
    # 

    def __init__(self, parent=None):
        super().__init__(parent)
        # Window settings
        self.setWindowTitle("PyPO")

        # init System
        self.stm = st.System(redirect=print, context="G")
        self.pyprint = print

        # GridParameters
        self.GPElementsColumn = [0, 0, 2, 1]
        self.GPParameterForm  = [0, 1, 2, 1]
        self.GPPlotScreen     = [0, 2, 1, 1]
        self.GPConsole        = [1, 2, 1, 1]

        
        # init layout
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(0)

        self._mkElementsColumn()
        self._setupPlotScreen()
        self._mkConsole()
        self.setLayout(self.grid)


        # NOTE Raytrace stuff
        self.frameDict = {}
        # end NOTE


    ##
    # Adds a widget to the layout of PyPOMainWidget
    def addToWindowGrid(self, widget, param):
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])

    ##
    # Generates a form widget
    # 
    # @param formData List of InputDescription objects
    # @param readAction Function to be called when forms ok-button is clicked
    # 
    def setForm(self, formData, readAction):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)
        self.ParameterWid = formGenerator.FormGenerator(formData, readAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        # self.ParameterWid.setContentsMargins(5,5,5,5)
        scroll = QScrollArea()
        scroll.setWidget(self.ParameterWid)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # scroll.border
        scroll.setWidgetResizable(True)
        scroll.setContentsMargins(0,0,0,0)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)
        self.addToWindowGrid(scroll,self.GPParameterForm)

    ##  
    # Configures the console widget
    # 
    def _mkConsole(self):
        self.console = ConsoleGenerator.get()
        self.addToWindowGrid(self.console, self.GPConsole)
        self.cursor = QTextCursor(self.console.document())
        
        global print ##TODO: Remove print redefinitions
        def print(s, end=''):
            if end == '\r':
                self.cursor.select(QTextCursor.LineUnderCursor)
                self.cursor.removeSelectedText()
                self.console.insertPlainText(str(s))
            else:
                self.console.appendPlainText(str(s))
            self.console.repaint()
        
        self.console.appendPlainText("********** PyPO Console **********")
        self.addToWindowGrid(self.console, self.GPConsole)
    
    ##
    # constructs the elements column on the left side of the screen
    # 
    def _mkElementsColumn(self):
        # delete if exists
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)
        # rebuild 
        self.ElementsColumn = Accordion()

        scroll = QScrollArea()
        scroll.setWidget(self.ElementsColumn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setContentsMargins(0,0,0,0)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(300)
        self.addToWindowGrid(scroll, self.GPElementsColumn)

    ##
    # Constructs the tab widget which will later hold plots
    def _setupPlotScreen(self):
        self.PlotWidget = QTabWidget()
        self.PlotWidget.setTabsClosable(True)
        self.PlotWidget.setTabShape(QTabWidget.Rounded)
        self.PlotWidget.tabCloseRequested.connect(self.closeTab)
        self.PlotWidget.setMaximumHeight(550)
        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)

    ##
    # TODO: doc
    def _formatVector(self, vector):
        return f"[{vector[0]}, {vector[1]}, {vector[2]}]"
    ##
    # Constructs a PlotScreen and adds it to the plotWidget along with a label for the tab
    def addPlot(self, figure, label):
        self.PlotWidget.addTab(PlotScreen(figure, parent=self), label)
        self.PlotWidget.setCurrentIndex(self.PlotWidget.count()-1)

    ##
    # removes a plot from the tabWidget
    # @param i Index of the plot to be removed
    def closeTab(self, i):
        self.PlotWidget.removeTab(i)

    ##
    # removes an element from the system
    # @param element Name of the element in the system
    def removeElement(self, element):
        print(f"removed: {element}")
        self.stm.removeElement(element)
    
    ##
    # plots a sigle element from the System
    # 
    # @param surface str Name of the surface in system
    # 
    def plotElement(self, surface):
        if self.stm.system:
            figure, _ = self.stm.plot3D(surface, show=False, save=False, ret=True)
        else :
            figure = None
        self.addPlot(figure, surface)

    ##
    # plots all elements of the system in one plot
    def plotSystem(self):
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False)
        else :
            figure = None
        self.addPlot(figure, "System Plot %d" %self.getSysPlotNr())
        

    ##
    # Gets the plot number froms System and increments it TODO: doc
    # @return The incremented plot number 
    def getSysPlotNr(self):
        if not hasattr(self, "sysPlotNr"):
            self.sysPlotNr = 0
        self.sysPlotNr+=1
        return self.sysPlotNr

    ##
    # Gets the frame plot number froms System and increments it TODO: doc
    # @return The incremented frame plot number 
    def getRayPlotNr(self):
        if not hasattr(self, "rayPlotNr"):
            self.rayPlotNr = 0
        self.rayPlotNr+=1
        return self.rayPlotNr

    ##
    # plots all elements of the system including ray traces in one plot
    def plotSystemWithRaytrace(self):
        framelist = []

        if self.stm.frames:
            for key in self.stm.frames.keys():
                framelist.append(key)
        
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False, RTframes=framelist)
        
        else:
            figure = None
        self.addPlot(figure,"Ray Trace Frame %d" %(self.getRayPlotNr()))

    ##
    # opens a form that allows user to save the System
    def saveSystemAction(self):
        self.setForm(fDataObj.saveSystemForm(), readAction=self.saveSystemCall)
    
    ##
    # opens a form that allows user to load a saved System
    def loadSystemAction(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fDataObj.loadSystemForm(systemList), readAction=self.loadSystemCall)
    
    ##
    # opens a form that allows user to delete a saved System
    def deleteSystemAction(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fDataObj.loadSystemForm(systemList), readAction=self.removeSystemCall)
    
    ##
    # Saves the current system state under the name given in form
    def saveSystemCall(self):
        saveDict = self.ParameterWid.read()
        self.stm.saveSystem(saveDict["name"]) 
    
    ##
    # Loads system selected in from form
    def loadSystemCall(self):
        loadDict = self.ParameterWid.read()
        self._mkElementsColumn()
        self.stm.loadSystem(loadDict["name"]) 
        self.refreshColumn(self.stm.system, "elements")
        self.refreshColumn(self.stm.frames, "frames")
        self.refreshColumn(self.stm.fields, "fields")
        self.refreshColumn(self.stm.currents, "currents")
        self.refreshColumn(self.stm.scalarfields, "scalarfields")

    ##
    # Deletes system selected in form
    def removeSystemCall(self):
        removeDict = self.ParameterWid.read()
        shutil.rmtree(os.path.join(self.stm.savePathSystems, removeDict["name"]))

    ##
    # TODO: @Maikel Rename this function and evaluate its nessecity
    def refreshColumn(self, columnDict, columnType):
        for key, item in columnDict.items():
            if columnType == "elements":
                self.ElementsColumn.reflectors.addWidget(ReflectorWidget(key, self.removeElement, self.setTransformationForm, self.plotElement))
            
            elif columnType == "frames":
                self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(key, self.stm.removeFrame, self.setPlotFrameFormOpt,  self.calcRMSfromFrame))
            
            elif columnType == "fields":
                self.ElementsColumn.POFields.addWidget(FieldsWidget(key,self.stm.removeField, self.setPlotFieldFormOpt))
            
            elif columnType == "currents":
                self.ElementsColumn.POCurrents.addWidget(CurrentWidget(key, self.stm.removeCurrent, self.setPlotFieldFormOpt))

            elif columnType == "scalarfields":
                self.ElementsColumn.SPOFields.addWidget(SFieldsWidget(key,self.stm.removeScalarField, self.setPlotSFieldFormOpt))

    ##
    # Shows from to add a quadric surface
    def setQuadricForm(self):
        self.setForm(fDataObj.makeQuadricSurfaceInp(), readAction=self.addQuadricAction)
    
    ##
    # Shows form to add a plane
    def setPlaneForm(self):
        self.setForm(fDataObj.makePlaneInp(), readAction=self.addPlaneAction)

    ##
    # Reads quadric form, evaluates surface type and calls corresponding add____Action
    def addQuadricAction(self):
        try:
            elementDict = self.ParameterWid.read()
            if elementDict["type"] == "Parabola":
                self.stm.addParabola(elementDict)
            elif elementDict["type"] == "Hyperbola":
                self.stm.addHyperbola(elementDict)
            elif elementDict["type"] == "Ellipse":
                self.stm.addEllipse(elementDict)
            self.ElementsColumn.reflectors.addWidget(ReflectorWidget(elementDict["name"], self.removeElement, self.setTransformationForm, self.plotElement))
        except InputReflError as e: #TODO: Does this errorCatching work?
            self.console.appendPlainText("FormInput Incorrect:")
            self.console.appendPlainText(e.__str__())

    ##
    # Reads form and adds parabola to System
    def addParabolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addParabola(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ReflectorWidget(elementDict["name"],self.removeElement, self.setTransformationForm, self.plotElement))
    
    ##
    # Reads form and adds hyperbola to System
    def addHyperbolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addHyperbola(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ReflectorWidget(elementDict["name"],self.removeElement, self.setTransformationForm, self.plotElement))

    ##
    # Reads form and adds ellipse to System
    def addEllipseAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addEllipse(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ReflectorWidget(elementDict["name"],self.removeElement, self.setTransformationForm, self.plotElement))
    
    ##
    # Reads form and adds plane to System
    def addPlaneAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addPlane(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ReflectorWidget(elementDict["name"],self.removeElement,self.setTransformationForm,self.plotElement))
    
    ##
    # Shows single element transformation form
    def setTransformationForm(self, element):
        self.setForm(fDataObj.makeTransformationForm(element), self.applyTransformation)

    ##
    # Shows multiple element transformation form
    def setTransformationElementsForm(self):
        movableElements = []
        for key, elem in self.stm.system.items():
            if elem["gmode"] != 2:
                movableElements.append(key)

        self.setForm(
            [InputDescription(inType.elementSelector, "elements", options=movableElements)]+
            fDataObj.makeTransformationElementsForm(self.stm.system.keys()), self.applyTransformationElements
            )
    
    ##
    # Applies single element transformation
    def applyTransformation(self, element):
        dd = self.ParameterWid.read()
        transformationType = dd["type"]
        vector = dd["vector"]

        if transformationType == "Translation":
            self.stm.translateGrids(dd["element"], vector)
            print(f'Translated {dd["element"]} by {self._formatVector(vector)} mm')
        elif transformationType == "Rotation":
            self.stm.rotateGrids(dd["element"], vector, cRot=dd["pivot"])
            print(f'Rotated {dd["element"]} by {self._formatVector(vector)} deg around {self._formatVector(dd["centerOfRotation"])}')
        else:
            raise Exception("Transformation type incorrect")

    ##
    # Applies multiple element transformation
    def applyTransformationElements(self):
        transfDict = self.ParameterWid.read()

        if transfDict["type"] == "Translation":
            self.stm.translateGrids(transfDict["elements"], transfDict["vector"])
            print(f'Translated {transfDict["elements"]} by {self._formatVector(transfDict["vector"])} mm')

        if transfDict["type"] == "Rotation":
            self.stm.rotateGrids(transfDict["elements"], transfDict["vector"], transfDict["pivot"])
            print(f'Translated {transfDict["elements"]} by {self._formatVector(transfDict["vector"])} mm')
    
    #NOTE Raytrace widgets

    ##
    # Shows tube frame form
    def setInitTubeFrameForm(self):
        self.setForm(fDataObj.initTubeFrameInp(), readAction=self.addTubeFrameAction)
    
    ##
    # Shows gaussian frame form
    def setInitGaussianFrameForm(self):
        self.setForm(fDataObj.initGaussianFrameInp(), readAction=self.addGaussianFrameAction)
    
    ##
    # Shows gaussian TODO: gaussian beam? form 
    def setInitGaussianForm(self):
        self.setForm(fDataObj.initGaussianInp(self.stm.system), readAction=self.addGaussianAction)
    
    ##
    # Shows gaussian TODO: gaussian s? form 
    def setInitSGaussianForm(self):
        self.setForm(fDataObj.initSGaussianInp(self.stm.system), readAction=self.addSGaussianAction)
    
    ##
    # TODO:doc
    def setInitPSForm(self):
        self.setForm(fDataObj.initPSInp(self.stm.system), readAction=self.addPSAction)
    
    ##
    # TODO:doc
    def setInitSPSForm(self):
        self.setForm(fDataObj.initSPSInp(self.stm.system), readAction=self.addSPSAction)
    
    def addTubeFrameAction(self):
        RTDict = self.ParameterWid.read()
        self.stm.createTubeFrame(RTDict)
        self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(RTDict["name"],
                           self.stm.removeFrame,  self.setPlotFrameFormOpt, self.calcRMSfromFrame))    
    
    def addGaussianFrameAction(self):
        GRTDict = self.ParameterWid.read()

        if not "seed" in GRTDict.keys():
            GRTDict["seed"] = -1

        self.stm.createGRTFrame(GRTDict)
        self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(GRTDict["name"],
                             self.stm.removeFrame,self.setPlotFrameFormOpt, self.calcRMSfromFrame))    
    
    def addGaussianAction(self):
        GDict = self.ParameterWid.read()
        self.stm.createGaussian(GDict, GDict["surface"])
        self.ElementsColumn.POFields.addWidget(FieldsWidget(GDict["name"], self.stm.removeField, self.setPlotFieldFormOpt))
        self.ElementsColumn.POCurrents.addWidget(CurrentWidget(GDict["name"], self.stm.removeCurrent, self.setPlotCurrentFormOpt))
    
    def addSGaussianAction(self):
        GDict = self.ParameterWid.read()
        self.stm.createScalarGaussian(GDict, GDict["surface"])
        self.ElementsColumn.SPOFields.addWidget(SFieldsWidget(GDict["name"], self.stm.removeScalarField, self.setPlotSFieldFormOpt))
    
    def addPSAction(self):
        PSDict = self.ParameterWid.read()
        self.stm.generatePointSource(PSDict, PSDict["surface"])
        self.ElementsColumn.POFields.addWidget(FieldsWidget(PSDict["name"], self.stm.removeField, self.setPlotFieldFormOpt))
        self.ElementsColumn.POCurrents.addWidget(CurrentWidget(PSDict["name"], self.stm.removeCurrent, self.setPlotCurrentFormOpt))
    
    def addSPSAction(self):
        SPSDict = self.ParameterWid.read()
        self.stm.generatePointSourceScalar(SPSDict, SPSDict["surface"])
        self.ElementsColumn.SPOFields.addWidget(SFieldsWidget(SPSDict["name"], self.stm.removeScalarField, self.setPlotSFieldFormOpt))
    
    def setPlotFrameForm(self):
        self.setForm(fDataObj.plotFrameInp(self.stm.frames), readAction=self.addPlotFrameAction)
    
    def setPlotFrameFormOpt(self, frame):
        self.setForm(fDataObj.plotFrameOpt(frame), readAction=self.addPlotFrameAction)

    def setPlotFieldFormOpt(self, field):
        if self.stm.system[self.stm.fields[field].surf]["gmode"] == 2:
            self.setForm(fDataObj.plotFarField(field), readAction=self.addPlotFieldAction)
        else:
            self.setForm(fDataObj.plotField(field), readAction=self.addPlotFieldAction)
                
    def setPlotSFieldFormOpt(self, field):
        self.setForm(fDataObj.plotSField(field, self.stm.system[self.stm.scalarfields[field].surf]["gmode"]), readAction=self.addPlotSFieldAction)
    
    def setPlotCurrentFormOpt(self, current):
        self.setForm(fDataObj.plotCurrentOpt(current), readAction=self.addPlotCurrentAction)
    
    def addPlotFrameAction(self):
        plotFrameDict = self.ParameterWid.read()
        fig = self.stm.plotRTframe(plotFrameDict["frame"], project=plotFrameDict["project"], ret=True)
        self.addPlot(fig, f'{plotFrameDict["frame"]} - {plotFrameDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)

    def addPlotFieldAction(self):
        plotFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], plotFieldDict["comp"], 
                                    project=plotFieldDict["project"], ret=True)
        self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)
    
    def addPlotSFieldAction(self):
        plotSFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotSFieldDict["field"], 
                                    project=plotSFieldDict["project"], ret=True)
        self.addPlot(fig, f'{plotSFieldDict["field"]} - {plotSFieldDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)
    
    def addPlotCurrentAction(self):
        plotFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], 
                                    plotFieldDict["comp"], project=plotFieldDict["project"], ret=True)
        self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)
    
    def setPropRaysForm(self):
        self.setForm(fDataObj.propRaysInp(self.stm.frames, self.stm.system), self.addPropRaysAction)

    def addPropRaysAction(self): 
        propRaysDict = self.ParameterWid.read()
        self.stm.runRayTracer(propRaysDict)
        self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(propRaysDict["frame_out"], 
                                [self.setPlotFrameFormOpt, self.stm.removeFrame, self.calcRMSfromFrame]))
    
    def setPOInitForm(self):
        self.setForm(fDataObj.propPOInp(self.stm.currents, self.stm.scalarfields, self.stm.system), self.addPropBeamAction)
    
    def setPOFFInitForm(self):
        self.setForm(fDataObj.propPOFFInp(self.stm.currents, self.stm.system), self.addPropBeamAction)
    
    def setTaperEffsForm(self):
        self.setForm(fDataObj.calcTaperEff(self.stm.fields, self.stm.system), self.calcTaperAction)
    
    def setSpillEffsForm(self):
        self.setForm(fDataObj.calcSpillEff(self.stm.fields, self.stm.system), self.calcSpillAction)

    def setXpolEffsForm(self):
        self.setForm(fDataObj.calcXpolEff(self.stm.fields, self.stm.system), self.calcXpolAction)

    def setMBEffsForm(self):
        self.setForm(fDataObj.calcMBEff(self.stm.fields, self.stm.system), self.calcMBAction)
    
    def calcTaperAction(self):
        TaperDict = self.ParameterWid.read()
        eff_taper = self.stm.calcTaper(TaperDict["f_name"], TaperDict["comp"])
        print(f'Taper efficiency of {TaperDict["f_name"]}, component {TaperDict["comp"]} = {eff_taper}\n')
    
    def calcSpillAction(self):
        SpillDict = self.ParameterWid.read()

        aperDict = {
                "center"    : SpillDict["center"],
                "inner"      : SpillDict["inner"],
                "outer"      : SpillDict["outer"]
                }

        eff_spill = self.stm.calcSpillover(SpillDict["f_name"], SpillDict["comp"], aperDict)
        print(f'Spillover efficiency of {SpillDict["f_name"]}, component {SpillDict["comp"]} = {eff_spill}\n')
    
    def calcXpolAction(self):
        XpolDict = self.ParameterWid.read()
        eff_Xpol = self.stm.calcXpol(XpolDict["f_name"], XpolDict["co_comp"], XpolDict["cr_comp"])
        print(f'X-pol efficiency of {XpolDict["f_name"]}, co-component {XpolDict["co_comp"]} and X-component {XpolDict["cr_comp"]} = {eff_Xpol}\n')

    def calcMBAction(self):
        MBDict = self.ParameterWid.read()
        eff_mb = self.stm.calcMainBeam(MBDict["f_name"], MBDict["comp"], MBDict["thres"], MBDict["mode"])
        print(f'Main beam efficiency of {MBDict["f_name"]}, component {MBDict["comp"]} = {eff_mb}\n')
        self.ElementsColumn.SPOFields.addWidget(SFieldsWidget(f"fitGauss_{MBDict['f_name']}", self.stm.removeScalarField, self.setPlotSFieldFormOpt))

    def addPropBeamAction(self):
        propBeamDict = self.ParameterWid.read()
      
        if propBeamDict["mode"] == "scalar":
            subStr = "scalar field"
        else:
            subStr = propBeamDict["mode"]

        dialStr = f"Calculating {subStr} on {propBeamDict['t_name']}..."

        dial = SymDialog(dialStr)

        self.mgr = TManager.Manager("G", callback=dial.accept)
        t = self.mgr.new_gthread(target=self.stm.runPO, args=(propBeamDict,), calc_type=propBeamDict["mode"])
        
        dial.setThread(t)

        if dial.exec_():
            if propBeamDict["mode"] == "JM":
                self.ElementsColumn.POCurrents.addWidget(CurrentWidget(propBeamDict["name_JM"], self.stm.removeCurrent, self.setPlotCurrentFormOpt))
        
            elif propBeamDict["mode"] == "EH" or propBeamDict["mode"] == "FF":
                self.ElementsColumn.POFields.addWidget(FieldsWidget(propBeamDict["name_EH"], self.stm.removeField, self.setPlotFieldFormOpt))
        
            elif propBeamDict["mode"] == "JMEH":
                self.ElementsColumn.POCurrents.addWidget(CurrentWidget(propBeamDict["name_JM"], self.stm.removeCurrent, self.setPlotCurrentFormOpt))
                self.ElementsColumn.POFields.addWidget(FieldsWidget(propBeamDict["name_EH"], self.stm.removeField, self.setPlotFieldFormOpt))
        
            elif propBeamDict["mode"] == "EHP":
                self.ElementsColumn.POFields.addWidget(FieldsWidget(propBeamDict["name_EH"], self.stm.removeField, self.setPlotFieldFormOpt))
                self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(propBeamDict["name_P"], 
                                self.stm.removeFrame, self.setPlotFrameFormOpt,self.calcRMSfromFrame))
    
            elif propBeamDict["mode"] == "scalar":
                self.ElementsColumn.SPOFields.addWidget(SFieldsWidget(propBeamDict["name_field"], self.stm.removeScalarField, self.setPlotSFieldFormOpt))

    #END NOTE
    
    def calcRMSfromFrame(self, frame):
        rms = self.stm.calcSpotRMS(frame)
        print(f"RMS value of {frame} = {rms} mm\n")

class PyPOMainWindow(QMainWindow):
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.mainWid = MainWidget()
        self.mainWid.setContentsMargins(0,0,0,0)
        self.setContentsMargins(0,0,0,0)
        self.setAutoFillBackground(True)
        self._createMenuBar()
        self.setCentralWidget(self.mainWid)
        self.showMaximized()
        with open('src/GUI/style.css') as f:
            style = f.read()
        self.setStyleSheet(style)


    def _createMenuBar(self):
        menuBar = self.menuBar()

        ElementsMenu    = menuBar.addMenu("Elements")
        SystemsMenu     = menuBar.addMenu("Systems")
        RaytraceMenu    = menuBar.addMenu("Ray-tracer")
        PhysOptMenu     = menuBar.addMenu("Physical-optics")
        ToolsMenu       = menuBar.addMenu("Tools")

        # ### Generate test parabola
        # AddTestParabola = QAction('Add Test Parabola', self)
        # AddTestParabola.setShortcut('Ctrl+Shift+P')
        # AddTestParabola.setStatusTip('Generates a Parabolic reflector and plots it')
        # AddTestParabola.triggered.connect(self.mainWid.addExampleParabola)
        # ElementsMenu.addAction(AddTestParabola)

        # ### Generate test hyperbola
        # AddTestHyperbola = QAction('Add Test Hyperbola', self)
        # AddTestHyperbola.setShortcut('Ctrl+Shift+H')
        # AddTestHyperbola.setStatusTip('Generates a Parabolic reflector and plots it')
        # AddTestHyperbola.triggered.connect(self.mainWid.addExampleHyperbola)
        # ElementsMenu.addAction(AddTestHyperbola)

        ### Add Element
        reflectorSelector = ElementsMenu.addMenu("Reflector")
        ### Planar Surface
        planeAction = QAction("Plane", self)
        planeAction.setShortcut("Ctrl+L")
        planeAction.setStatusTip("Add a planar element.")
        planeAction.triggered.connect(self.mainWid.setPlaneForm)
        reflectorSelector.addAction(planeAction)
        
        ### Quadric Surface
        hyperbolaAction = QAction('Quadric surface', self)
        hyperbolaAction.setShortcut('Ctrl+Q')
        hyperbolaAction.setStatusTip("Add a paraboloid, hyperboloid or ellipsoid element.")
        hyperbolaAction.triggered.connect(self.mainWid.setQuadricForm)
        reflectorSelector.addAction(hyperbolaAction)

        transformElementsAction = QAction("Transform elements", self)
        transformElementsAction.setStatusTip("Transform a group of elements.")
        transformElementsAction.triggered.connect(self.mainWid.setTransformationElementsForm)
        ElementsMenu.addAction(transformElementsAction)
        

    ### System actions
        # newSystem = QAction('Add System', self)
        # newSystem.triggered.connect(self.mainWid.addSystemAction)
        # SystemsMenu.addAction(newSystem)

        plotSystem = QAction("Plot System", self)
        plotSystem.setStatusTip("Plot all elements in the current system.")
        plotSystem.triggered.connect(self.mainWid.plotSystem)
        SystemsMenu.addAction(plotSystem)

        plotRaytrace = QAction("Plot ray-trace", self)
        plotSystem.setStatusTip("Plot all elements in the current system, including ray-traces.")
        plotRaytrace.triggered.connect(self.mainWid.plotSystemWithRaytrace)
        SystemsMenu.addAction(plotRaytrace)
        
        saveSystem = QAction("Save system", self)
        saveSystem.setStatusTip("Save the current system to disk.")
        saveSystem.triggered.connect(self.mainWid.saveSystemAction)
        SystemsMenu.addAction(saveSystem)

        loadSystem = QAction("Load system", self)
        loadSystem.setStatusTip("Load a saved system from disk.")
        loadSystem.triggered.connect(self.mainWid.loadSystemAction)
        SystemsMenu.addAction(loadSystem)
        
        removeSystem = QAction("Remove system", self)
        removeSystem.setStatusTip("Remove a saved system from disk.")
        removeSystem.triggered.connect(self.mainWid.deleteSystemAction)
        SystemsMenu.addAction(removeSystem)
        
        # NOTE Raytrace actions
        makeFrame = RaytraceMenu.addMenu("Make frame")
        initTubeFrameAction = QAction("Tube", self)
        initTubeFrameAction.setStatusTip("Initialize ray-trace tube from input form")
        initTubeFrameAction.triggered.connect(self.mainWid.setInitTubeFrameForm)
        makeFrame.addAction(initTubeFrameAction)

        initGaussianFrameAction = QAction("Gaussian", self)
        initGaussianFrameAction.setStatusTip("Initialize ray-trace Gaussian from input form")
        initGaussianFrameAction.triggered.connect(self.mainWid.setInitGaussianFrameForm)
        makeFrame.addAction(initGaussianFrameAction)
        
        # Propagate rays
        propRaysAction = QAction("Propagate rays", self)
        propRaysAction.setStatusTip("Propagate a frame of rays to a target surface")
        propRaysAction.triggered.connect(self.mainWid.setPropRaysForm)
        RaytraceMenu.addAction(propRaysAction)
        
        # PO actions
        makeBeam = PhysOptMenu.addMenu("Initialize beam")
        makeBeamPS = makeBeam.addMenu("Point source")
        initPointVecAction = QAction("Vectorial", self)
        initPointVecAction.setStatusTip("Initialize a vectorial point source.")
        initPointVecAction.triggered.connect(self.mainWid.setInitPSForm)
        makeBeamPS.addAction(initPointVecAction)
        
        initPointScalAction = QAction("Scalar", self)
        initPointScalAction.setStatusTip("Initialize a scalar point source.")
        initPointScalAction.triggered.connect(self.mainWid.setInitSPSForm)
        makeBeamPS.addAction(initPointScalAction)
    
        makeBeamG = makeBeam.addMenu("Gaussian beam")
        initGaussVecAction = QAction("Vectorial", self)
        initGaussVecAction.setStatusTip("Initialize a vectorial Gaussian beam.")
        initGaussVecAction.triggered.connect(self.mainWid.setInitGaussianForm)
        makeBeamG.addAction(initGaussVecAction)
        
        initGaussScalAction = QAction("Scalar", self)
        initGaussScalAction.setStatusTip("Initialize a scalar Gaussian beam.")
        initGaussScalAction.triggered.connect(self.mainWid.setInitSGaussianForm)
        makeBeamG.addAction(initGaussScalAction)

        propBeam = PhysOptMenu.addMenu("Propagate beam") 
        initPropSurfAction = QAction("To surface", self)
        initPropSurfAction.setStatusTip("Propagate a PO beam from a source surface to a target surface.")
        initPropSurfAction.triggered.connect(self.mainWid.setPOInitForm)
        propBeam.addAction(initPropSurfAction)
        
        initPropFFAction = QAction("To far-field", self)
        initPropSurfAction.setStatusTip("Propagate a PO beam from a source surface to a far-field surface.")
        initPropFFAction.triggered.connect(self.mainWid.setPOFFInitForm)
        propBeam.addAction(initPropFFAction)

        calcEffs = PhysOptMenu.addMenu("Efficiencies")
        calcSpillEffsAction = QAction("Spillover", self)
        calcSpillEffsAction.setStatusTip("Calculate spillover efficiency of a PO field.")
        calcSpillEffsAction.triggered.connect(self.mainWid.setSpillEffsForm)
        calcEffs.addAction(calcSpillEffsAction)
        
        calcTaperEffsAction = QAction("Taper", self)
        calcTaperEffsAction.setStatusTip("Calculate taper efficiency of a PO field.")
        calcTaperEffsAction.triggered.connect(self.mainWid.setTaperEffsForm)
        calcEffs.addAction(calcTaperEffsAction)
        
        calcXpolEffsAction = QAction("X-pol", self)
        calcXpolEffsAction.setStatusTip("Calculate cross-polar efficiency of a PO field.")
        calcXpolEffsAction.triggered.connect(self.mainWid.setXpolEffsForm)
        calcEffs.addAction(calcXpolEffsAction)

        calcMBEffsAction = QAction("Main beam", self)
        calcMBEffsAction.setStatusTip("Calculate main beam efficiency of a PO field.")
        calcMBEffsAction.triggered.connect(self.mainWid.setMBEffsForm)
        calcEffs.addAction(calcMBEffsAction)

        findRTfocusAction = QAction("Find ray-trace focus", self)
        findRTfocusAction.setStatusTip("Calculate the focus co-ordinates of a ray-trace beam.")
        #findRTfocusAction.triggered.connect(self.mainWid.set)

if __name__ == "__main__":

    print("lala")
    app = QApplication(sys.argv)
    win = PyPOMainWindow(parent=None)
    # def print(s):
    #     cons.appendPlainText(s)
    win.show()
    app.exec_()
