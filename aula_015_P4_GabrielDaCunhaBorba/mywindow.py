from typing import Callable, Dict
from time import localtime
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from mycanvas import *
from mymodel import *
from simulationParameters import SimulationParameters


class MyWindow(QMainWindow):
    '''
    Classe responsável por conhecer um grupo de widgets que irão se comunicar.
    '''

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle('MyGlDrawer')

        # NOTE - create a canvas object to be used by opengl
        self.canvas: MyCanvas = MyCanvas()
        self.setCentralWidget(self.canvas)

        # NOTE - create a model object and pass to canvas
        self.model: MyModel = MyModel()
        self.gridOnModel: MyCloudPoint = MyCloudPoint()
        self.canvas.setModel(self.model)

        # NOTE - create a model object (from hetool) and pass to canvas
        self.hemodel = self.canvas.hemodel = HeModel()
        self.heview = self.canvas.heview = HeView(self.hemodel)
        self.hecontroller = self.canvas.hecontroller = HeController(self.hemodel)
        self.hetolerance = self.canvas.hetolerance = 0.01

        # NOTE - create the dialog to prompt the user about the size of the cloud point
        self.modelPointCloudDialog: ModelPointCloudDialog = ModelPointCloudDialog()
        # NOTE - create the point cloud when the dialog become accepted
        self.modelPointCloudDialog.accepted.connect(self.__createShowPointCloud)
        # NOTE - create the dialog to prompt the user about the boundary conditions of the cloud point
        self.modelBCDialog: ModelBCDialog = ModelBCDialog()
        # NOTE - apply the bc on the selected points when the dialog become accepted
        self.modelBCDialog.accepted.connect(self.__applyBCSelectedPoints)
        self.modelBCDialog.finished.connect(self.modelBCDialog.resetDialog)

        # NOTE - making a toolbar
        tb = self.addToolBar("File")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
        tb2 = self.addToolBar("Simulation")
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, tb2)
        # NOTE - maing the toolbar actions

        tbActions: dict[QAction, Callable] = {
            QAction("Clear",self): self.__actionClearModel,
            QAction("Pan_L",self): self.__actionPanL,
            QAction("Pan_R",self): self.__actionPanR,
            QAction("Pan_T",self): self.__actionPanT,
            QAction("Pan_B",self): self.__actionPanB,
            QAction("Fit",self): self.__actionFit,
            QAction("Zoom In",self): self.__actionZoomIn,
            QAction("Zoom Out",self): self.__actionZoomOut
        }
        tb2Actions: dict[QAction, Callable]  = {
            QAction("Point Cloud",self): self.__actionOpenModelPointCloudDialog,
            QAction("Boundary condition",self): self.__actionOpenModelBCDialog,
            QAction("Export Parameters",self): self.__actionExportSimulationParameters
        }
        # NOTE  - connect signals and slots for the actions
        # LINK - https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#connecting-signals-using-keyword-arguments

        for a, f in tbActions.items():
            a.triggered.connect(f)
        # NOTE - adding the actions on the toolbar
        tb.addActions(tbActions.keys())
        # tb.actionTriggered[QAction].connect(self.tbpressed)

        for a, f in tb2Actions.items():
            a.triggered.connect(f)
        tb2.addActions(tb2Actions.keys())

    def __actionClearModel(self):
        self.model.clear()
        self.hemodel = self.canvas.hemodel = HeModel()
        self.heview = self.canvas.heview = HeView(self.hemodel)
        self.hecontroller = self.canvas.hecontroller = HeController(self.hemodel)
        self.gridOnModel = MyCloudPoint()
        self.canvas.hepointCloud.clear()
        self.canvas.selectedPoints.clear()
        self.__refreshCanvas()

    def __actionFit(self):
        self.canvas.fitWorldToViewport()
        self.canvas.paintGL()

    def __actionPanL(self):
        self.canvas.panWorldWindow(0.1, 0.0)
        self.canvas.paintGL()

    def __actionPanR(self):
        self.canvas.panWorldWindow(-0.1, 0.0)
        self.canvas.paintGL()

    def __actionPanB(self):
        self.canvas.panWorldWindow(0.0, -0.1)
        self.canvas.paintGL()

    def __actionPanT(self):
        self.canvas.panWorldWindow(0.0, 0.1)
        self.canvas.paintGL()

    def __actionZoomIn(self):
        self.canvas.scaleWorldWindow(0.90)
        self.__refreshCanvas()

    def __actionZoomOut(self):
        self.canvas.scaleWorldWindow(1.10)
        self.__refreshCanvas()

    def __actionOpenModelPointCloudDialog(self):
        # LINK - https://doc.qt.io/qt-5/qdialog.html#exec
        self.modelPointCloudDialog.open()

    def __actionOpenModelBCDialog(self):
        self.modelBCDialog.open()

    def __actionExportSimulationParameters(self):
        simParam = SimulationParameters()
        simParam.x_dist = self.gridOnModel.xSpacing
        simParam.y_dist = self.gridOnModel.ySpacing
        simParam.ij_pos = [e.position for e in self.gridOnModel.entries]
        simParam.connect = [e.neighbors for e in self.gridOnModel.entries]
        simParam.boundary_condition = [(1 if e.hasDirichletBC else 0, e.boundaryConditionValue) for e in self.gridOnModel.entries]
        prefix = './output/'
        timeNow = localtime()
        filename = f'{timeNow.tm_year:04}y{timeNow.tm_yday:03}d{timeNow.tm_hour:02}h{timeNow.tm_min:02}m{timeNow.tm_sec:02}sIn'
        simParam.saveAsJson(f'{prefix}{filename}.json')

    def __createShowPointCloud(self):
        # NOTE - create grid
        self.gridOnModel.fromDesiredNumberOfDivisions(
            self.heview,
            self.modelPointCloudDialog.desiredX,
            self.modelPointCloudDialog.desiredY)
        self.canvas.selectedPoints.clear()
        self.canvas.hepointCloud = [e.coord for e in self.gridOnModel.entries]
        # NOTE - show grid
        self.__refreshCanvas()

    def __refreshCanvas(self):
        self.canvas.update()
        self.canvas.paintGL()

    def __applyBCSelectedPoints(self):  # FIXME
        for e in self.gridOnModel.entries:
            if e.coord in self.canvas.selectedPoints:
                e.hasDirichletBC = self.modelBCDialog.initialConditionActive
                e.boundaryConditionValue = self.modelBCDialog.initialConditionValue
        self.canvas.selectedPoints.clear()


class ModelPointCloudDialog(QDialog):  # FIXME
    def __init__(self) -> None:
        super().__init__()
        # NOTE - creating the input fields for integers
        self._xSpinBox: QSpinBox = QSpinBox()
        self._xSpinBox.setRange(0, 10000)
        self._xSpinBox.setSingleStep(1)
        self._ySpinBox: QSpinBox = QSpinBox()
        self._ySpinBox.setRange(0, 10000)
        self._ySpinBox.setSingleStep(1)

        # NOTE - helper text
        t2 = QLabel('Obs.: A nuvem será limpa se nenhuma subdivisão for definida.')
        t2.setWordWrap(True)

        group1 = QGroupBox('Subdivisões por eixo')
        # NOTE - create a layout with the labels and input fields
        form1 = QFormLayout()
        form1.addRow(QLabel('x'), self._xSpinBox)
        form1.addRow(QLabel('y'), self._ySpinBox)
        group1.setLayout(form1)

        # NOTE - create the bottom bar with buttons for the dialog
        buttonBox = QDialogButtonBox()
        buttonBox\
            .addButton(QDialogButtonBox.StandardButton.Apply)\
            .clicked.connect(self.accept)
        buttonBox\
            .addButton(QDialogButtonBox.StandardButton.Cancel)\
            .clicked.connect(self.reject)

        # NOTE - layout that wraps up all the widget for the dialog
        layout = QVBoxLayout()
        layout.addWidget(group1)
        layout.addWidget(t2)
        layout.addWidget(buttonBox)

        # NOTE - configures the dialog
        self.setWindowTitle('Nuvem de pontos')
        self.setLayout(layout)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

    @property
    def desiredX(self):
        return self._xSpinBox.value()

    @property
    def desiredY(self):
        return self._ySpinBox.value()


class ModelBCDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self._bcEnabledCheckBox: QCheckBox = QCheckBox()
        self._bcEnabledCheckBox.setText('Habilitar condição imposta.')
        self._bcEnabledCheckBox.setChecked(False)
        self._bcEnabledCheckBox.clicked.connect(self._updateSpinBoxState)
        self._bcValueLine: QDoubleSpinBox = QDoubleSpinBox()
        self._bcValueLine.setRange(-999999999999.0, 999999999999.0)
        self._bcValueLine.setSingleStep(1.0)
        self._bcValueLine.setDecimals(3)
        self._updateSpinBoxState()

        # NOTE - helper text
        t2 = QLabel('Ao aplicar, essas configurações serão aplicadas aos pontos que estiverem atualmente selecionados.')
        t2.setWordWrap(True)

        group1 = QGroupBox()
        # NOTE - create a layout with the labels and input fields
        form1 = QFormLayout()
        form1.addRow(self._bcEnabledCheckBox)
        form1.addRow(QLabel('Temperatura (°C):'), self._bcValueLine)
        group1.setLayout(form1)

        # NOTE - create the bottom bar with buttons for the dialog
        buttonBox = QDialogButtonBox()
        buttonBox\
            .addButton(QDialogButtonBox.StandardButton.Apply)\
            .clicked.connect(self.accept)
        buttonBox\
            .addButton(QDialogButtonBox.StandardButton.Cancel)\
            .clicked.connect(self.reject)

        # NOTE - layout that wraps up all the widget for the dialog
        layout = QVBoxLayout()
        layout.addWidget(group1)
        layout.addWidget(t2)
        layout.addWidget(buttonBox)

        # NOTE - configures the dialog
        self.setWindowTitle('Impor condição')
        self.setLayout(layout)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

    @property
    def initialConditionActive(self):
        return self._bcEnabledCheckBox.isChecked()

    @property
    def initialConditionValue(self):
        return self._bcValueLine.value()

    def resetDialog(self):
        self._bcEnabledCheckBox.setChecked(False)
        self._updateSpinBoxState()
        self._bcValueLine.setValue(0)

    def _updateSpinBoxState(self):
        self._bcValueLine.setEnabled(self._bcEnabledCheckBox.isChecked())
