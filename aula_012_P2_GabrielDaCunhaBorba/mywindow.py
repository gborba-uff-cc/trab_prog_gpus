from typing import Callable, Dict
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import json

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
        self.canvas.setModel(self.model)

        # NOTE - create a model object (from hetool) and pass to canvas
        self.hemodel = self.canvas.hemodel = HeModel()
        self.heview = self.canvas.heview = HeView(self.hemodel)
        self.hecontroller = self.canvas.hecontroller = HeController(self.hemodel)
        self.hetolerance = self.canvas.hetolerance = 0.01

        # NOTE - create the dialog to prompt the user about the size of the cloud point
        self.modelPointCloudDialog: ModelPointCloudDialog = ModelPointCloudDialog()
        # NOTE - create the point cloud when the dialog become accepted
        self.modelPointCloudDialog.accepted.connect(self.__createShowSavePointCloud)

        # NOTE - create the model point cloud collection
        self.pointCloud: List[Tuple[float, float]] = []

        # NOTE - making a toolbar
        tb = self.addToolBar("File")
        # NOTE - maing the toolbar actions
        self.__tbActions: dict[QAction, Callable] = {
            QAction("Clear",self): self.__actionClearModel,
            QAction("Fit",self): self.__actionFit,
            QAction("Pan_L",self): self.__actionPanL,
            QAction("Pan_R",self): self.__actionPanR,
            QAction("Pan_T",self): self.__actionPanT,
            QAction("Pan_B",self): self.__actionPanB,
            QAction("Zoom In",self): self.__actionZoomIn,
            QAction("Zoom Out",self): self.__actionZoomOut,
            QAction("Point Cloud",self): self.__actionOpenModelPointCloudDialog,
        }
        # NOTE  - connect signals and slots for the actions
        # LINK - https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#connecting-signals-using-keyword-arguments
        for a, f in self.__tbActions.items():
            a.triggered.connect(f)
        # NOTE - adding the actions on the toolbar
        tb.addActions(self.__tbActions.keys())
        # tb.actionTriggered[QAction].connect(self.tbpressed)

    def __actionClearModel(self):
        self.model.clear()
        self.hemodel = self.canvas.hemodel = HeModel()
        self.heview = self.canvas.heview = HeView(self.hemodel)
        self.hecontroller = self.canvas.hecontroller = HeController(self.hemodel)
        self.pointCloud = self.canvas.hepointCloud = []
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
        self.canvas.update()
        self.canvas.paintGL()

    def __actionZoomOut(self):
        self.canvas.scaleWorldWindow(1.10)
        self.canvas.update()
        self.canvas.paintGL()

    def __actionOpenModelPointCloudDialog(self):
        # LINK - https://doc.qt.io/qt-5/qdialog.html#exec
        self.modelPointCloudDialog.open()

    def __createShowSavePointCloud(self):
        try:
            # NOTE - create
            # pointCloud = self.__createPointCloud()
            simParam = self.__createPointCloud()
            pointCloud = simParam.particle_coords
            # NOTE - show
            self.pointCloud = self.canvas.hepointCloud = pointCloud
            # NOTE - save
            # self.__savePointCloud(pointCloud, 'model_point_cloud.json')
            bma = 0.05 - 0.0

            prefix = './output/q2in'
            simParam.step_size = 0.0005
            cloudSizeStr = f'{self.modelPointCloudDialog.desiredX}x{self.modelPointCloudDialog.desiredY}'
            simParam.number_steps = int(bma/simParam.step_size)
            simParam.saveAsJson(f'{prefix}_1_{cloudSizeStr}.json')
            simParam.step_size = 0.0001
            simParam.number_steps = int(bma/simParam.step_size)
            simParam.saveAsJson(f'{prefix}_2_{cloudSizeStr}.json')
            simParam.step_size = 0.00005
            simParam.number_steps = int(bma/simParam.step_size)
            simParam.saveAsJson(f'{prefix}_3_{cloudSizeStr}.json')

        except ValueError as ve:
            print(ve)
            self.__actionOpenModelPointCloudDialog()
        self.__refreshCanvas()

    def __createPointCloud(self):
        # NOTE - retrieve the user input from the dialog
        x: int = self.modelPointCloudDialog.desiredX
        y: int = self.modelPointCloudDialog.desiredY
        aux = self.__pointCloudGenerator(x, y)
        mass = self.modelPointCloudDialog.particleMass
        hardness = self.modelPointCloudDialog.particleHardness
        aux.particle_mass = mass
        aux.particle_hardness = hardness
        return aux

    def __refreshCanvas(self):
        self.canvas.update()
        self.canvas.paintGL()

    def __savePointCloud(self, pointCloud, archiveName):
        # NOTE - writing the point cloud on a file
        with open(archiveName, 'w') as writeFile:
            json.dump(pointCloud, writeFile)

    def __loadPointCloud(self, archiveName):
        # NOTE - reading the point cloud from a file
        content = []
        with open(archiveName, 'r') as readFile:
            content = json.load(readFile)
        return content

    # REVIEW - still waiting to find the correct place to put this method on
    def __pointCloudGenerator(self, desiredX, desiredY) -> SimulationParameters:
        def __fullNeighborhood(desiredX, desiredY) -> Dict[int, List[int]]:
            neighbors = {}
            for (x,y) in ((x,y) for x in range(desiredX + 1) for y in range(desiredY + 1)):
                # NOTE - x+y*(desiredX+1) is the nth elemennt beeing counted
                # NOTE - four is the maximum number of neighbors
                # NOTE - y is in the inner loop; y is faster
                # NOTE - element counting starting from 1
                nthElem = x*(desiredY+1)+y+1
                neighbors[nthElem] = []
                if x != 0:  # NOTE - not the first in line
                    neighbors[nthElem].append( nthElem - (desiredY+1))
                if y != 0:  # NOTE - not the first in column
                    neighbors[nthElem].append( nthElem - 1)
                if y != desiredY:  # NOTE - not the last in column
                    neighbors[nthElem].append( nthElem + 1)
                if x != desiredX:  # NOTE - not the last in line
                    neighbors[nthElem].append( nthElem + (desiredY+1))
            return neighbors

        def __removeFromNeighbors(neighbors,target) -> Dict[int, List[int]]:
            totalNeighbors = len(neighbors)
            removed = neighbors.pop(target,None)
            if not removed:
                return neighbors
            # NOTE - element count starting from 1
            for k in range(1,totalNeighbors+1):
                if k == target:
                    continue
                try:
                    neighbors[k].remove(target)
                except ValueError as ve:
                    pass
                for j in range(len(neighbors[k])):
                    if neighbors[k][j] > target:
                        neighbors[k][j] -= 1
                if k > target:
                    neighbors[k-1] = neighbors[k]
                if k == totalNeighbors:
                    neighbors.pop(k)
            return neighbors
        # ======================================================================

        # NOTE - treat the entries
        if desiredX * desiredY <= 0 or self.heview.isEmpty():
            return SimulationParameters()
        # NOTE - get the bounding box from the hemodel's view
        xmin, xmax, ymin, ymax = self.heview.getBoundBox()
        if not xmin or not xmax or not ymin or not ymax:
            raise TypeError(
                f'''mywindow.__actionCreatePoinCloud>>> xmin or xmax or ymin or ymax isn't float {xmin=}, {xmax=}, {ymin=}, {ymax=}''')
        cloudParams = SimulationParameters()
        # NOTE - retrieve model patches
        patches = self.heview.getPatches()
        tempPoint = Point()
        pointCloud = []
        # NOTE - divide the model boundbox in x and y subdivisions retrieved from dialog
        xDivisionSize = (xmax - xmin) / (desiredX)
        yDivisionSize = (ymax - ymin) / (desiredY)

        # NOTE - generating the  full list of neighbors
        neighbors = __fullNeighborhood(desiredX, desiredY)

        # NOTE - getting the points inside the patches
        # x = 0
        # y = 0
        # while x < desiredX +1:
        #     while y < desiredY +1:
        nthElem = 1
        for (x,y) in ((x,y) for x in range(desiredX + 1) for y in range(desiredY + 1)):
            # nthElem = x*(desiredY+1)+y+1
            # NOTE - element counting starting from 1
            tempPoint.setX(xmin + x * xDivisionSize)
            tempPoint.setY(ymin + y * yDivisionSize)
            # NOTE - insert point if inside the model
            for patch in patches:
                if patch.isPointInside(tempPoint):
                    pointCloud.append((tempPoint.getX(), tempPoint.getY()))
                    nthElem += 1
                else:
                    # nthElem = x*(desiredY+1)+y+1
                    neighbors = __removeFromNeighbors(neighbors, nthElem)
        # NOTE - parameterize the pointCloud
        cloudParams.particle_half_xSize = abs(xDivisionSize) / 2
        cloudParams.particle_half_ySize = abs(yDivisionSize) / 2
        cloudParams.particle_coords = pointCloud
        cloudParams.particle_restricted = [[0.0]*len(pointCloud[0])]*len(pointCloud)
        cloudParams.particle_external_force = [[0.0]*len(pointCloud[0])]*len(pointCloud)
        cloudParams.particle_connection = neighbors
        return cloudParams

class ModelPointCloudDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        # NOTE - creating the input fields for integers
        self._xSpinBox: QSpinBox = QSpinBox()
        self._xSpinBox.setRange(0, 10000)
        self._xSpinBox.setSingleStep(1)
        self._ySpinBox: QSpinBox = QSpinBox()
        self._ySpinBox.setRange(0, 10000)
        self._ySpinBox.setSingleStep(1)
        self._massSpinBox: QDoubleSpinBox = QDoubleSpinBox()
        self._massSpinBox.setRange(0, 99999.0)
        self._massSpinBox.setSingleStep(1.0)
        self._massSpinBox.setDecimals(3)
        self._hardnessSpinBox: QDoubleSpinBox = QDoubleSpinBox()
        self._hardnessSpinBox.setRange(0, 999999999999.0)
        self._hardnessSpinBox.setSingleStep(1.0)
        self._hardnessSpinBox.setDecimals(3)

        # NOTE - helper text
        # t1 = QLabel('Entre com o número de subdivisões no eixo x e no eixo y.')
        # t1.setWordWrap(True)
        t2 = QLabel('Obs.: A grade será limpa se nenhuma subdivisão for definida.')
        t2.setWordWrap(True)

        group1 = QGroupBox('Subdivisões por eixo')
        # NOTE - create a layout with the labels and input fields
        form1 = QFormLayout()
        form1.addRow(QLabel('x'), self._xSpinBox)
        form1.addRow(QLabel('y'), self._ySpinBox)
        group1.setLayout(form1)

        group2 = QGroupBox('Configuração das partículas')
        # NOTE - create a layout with the labels and input fields
        form2 = QFormLayout()
        form2.addRow(QLabel('Massa'), self._massSpinBox)
        form2.addRow(QLabel('Rigidez'), self._hardnessSpinBox)
        group2.setLayout(form2)

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
        # layout.addWidget(t1)
        layout.addWidget(group1)
        layout.addWidget(group2)
        layout.addWidget(t2)
        layout.addWidget(buttonBox)

        # NOTE - configures the dialog
        self.setWindowTitle('Grade de pontos')
        self.setLayout(layout)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

    @property
    def desiredX(self):
        return self._xSpinBox.value()

    @property
    def desiredY(self):
        return self._ySpinBox.value()

    @property
    def particleMass(self):
        return self._massSpinBox.value()

    @property
    def particleHardness(self):
        return self._hardnessSpinBox.value()
