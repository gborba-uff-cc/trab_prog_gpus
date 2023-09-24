from copyreg import constructor
from typing import Any, List, Tuple, Union
from hetool import HeView, Point
from simulationParameters import SimulationParameters


class MyPoint:

    def __init__(self, x=0.0, y=0.0) -> None:
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value


class MyCurve:
    def __init__(self, p1: MyPoint, p2: MyPoint) -> None:
        self.__p1 = p1
        self.__p2 = p2

    @property
    def p1(self):
        return self.__p1

    @p1.setter
    def p1(self, value):
        self.__p1 = value

    @property
    def p2(self):
        return self.__p2

    @p2.setter
    def p2(self, value):
        self.__p2 = value


class MyModel:

    def __init__(self) -> None:
        self.__verts: List[MyPoint] = []
        self.__curves: List[MyCurve] = []
        self.__MIN_SIZE = (0.0,10.0,0.0,10.0)

    @property
    def verts(self) -> List[MyPoint]:
        return self.__verts

    def addVert(self, x, y):
        self.__verts.append(MyPoint(x, y))

    @property
    def curves(self) -> List[MyCurve]:
        return self.__curves

    def addCurve(self, x1, y1, x2, y2):
        self.__curves.append(MyCurve(MyPoint(x1, y1), MyPoint(x2, y2)))

    def isEmpty(self) -> bool:
        return len(self.__verts) == 0 and len(self.__curves) == 0

    def clear(self):
        self.__verts = []
        self.__curves = []

    def getBoundBox(self) -> 'tuple[float, float, float, float]':
        if len(self.__verts) == 0 and len(self.__curves) == 0:
            return self.__MIN_SIZE

        xmin, xmax = (0.0, 0.0)
        ymin, ymax = (0.0, 0.0)

        if len(self.__verts) > 0:
            xmin = self.__verts[0].x
            xmax = xmin
            ymin = self.__verts[0].y
            ymax = ymin
            for i in range(1,len(self.__verts)):
                if self.__verts[i].x() < xmin:
                    xmin = self.__verts[i].x
                if self.__verts[i].x() > xmax:
                    xmax = self.__verts[i].x
                if self.__verts[i].y() < ymin:
                    ymin = self.__verts[i].y
                if self.__verts[i].y() > ymax:
                    ymax = self.__verts[i].y

        if len(self.__curves) > 0:
            if len(self.__verts) == 0:
                xmin = min(self.__curves[0].p1.x,self.__curves[0].p2.x)
                xmax = max(self.__curves[0].p1.x,self.__curves[0].p2.x)
                ymin = min(self.__curves[0].p1.y,self.__curves[0].p2.y)
                ymax = max(self.__curves[0].p1.y,self.__curves[0].p2.y)
            for i in range(1,len(self.__curves)):
                temp_xmin = min(self.__curves[i].p1.x,self.__curves[i].p2.x)
                temp_xmax = max(self.__curves[i].p1.x,self.__curves[i].p2.x)
                temp_ymin = min(self.__curves[i].p1.y,self.__curves[i].p2.y)
                temp_ymax = max(self.__curves[i].p1.y,self.__curves[i].p2.y)
                if temp_xmin < xmin:
                    xmin = temp_xmin
                if temp_xmax > xmax:
                    xmax = temp_xmax
                if temp_ymin < ymin:
                    ymin = temp_ymin
                if temp_ymax > ymax:
                    ymax = temp_ymax
        return (xmin,xmax,ymin,ymax)

        # empty_vertices = len(self.__verts) == 0
        # empty_curves = len(self.__curves) == 0
        # if empty_vertices and empty_curves:
        #     return self.__MIN_SIZE

        # xmin, xmax = (None, )*2
        # ymin, ymax = (None, )*2

        # # set min and max values if there is points in verts and if min and max values are None
        # if not empty_vertices and xmin is None:
        #     xmin, xmax = (self.__verts[0].x,)*2
        #     ymin, ymax = (self.__verts[0].y,)*2
        # for point in self.__verts:
        #     # the minimum and maximum when reading the first vertex
        #     if point.x < xmin:
        #         xmin = point.x
        #     elif point.x > xmax:
        #         xmax = point.x
        #     if point.y < ymin:
        #         ymin = point.y
        #     elif point.y > ymax:
        #         ymax = point.y

        # if not empty_curves and xmin is None:
        #     xmin, xmax = (self.__curves[0].p1.x,)*2
        #     ymin, ymax = (self.__curves[0].p1.y,)*2
        # for curve in self.__curves:
        #     for point in (curve.p1, curve.p2):
        #         if point.x < xmin:
        #             xmin = point.x
        #         elif point.x > xmax:
        #             xmax = point.x
        #         if point.y < ymin:
        #             ymin = point.y
        #         elif point.y > ymax:
        #             ymax = point.y

        # return (xmin, xmax, ymin, ymax)

# class MyCloudPointPoint:
#     def __init__(self) -> None:
#         self.hePoint = Point()
#         self.insideModel = False
#         self.globalId = 0
#         self.localId = 0

class MyCloudPointEntry:
    def __init__(self) -> None:
        self.coord: Tuple[Any,Any] = (0.0,0.0)
        self.position: Tuple[int,int] = (0,0)
        self.neighbors: Tuple[int,int,int,int] = (0,0,0,0)
        self.hasDirichletBC = False
        self.boundaryConditionValue = 0.0

class MyCloudPoint:
    def __init__(self) -> None:
        self.xSpacing = 0.0
        self.ySpacing = 0.0
        self.entries: List[MyCloudPointEntry] = []
        self.pointsInsideModel: List[Tuple[Any,Any]] = []
        self.pointsConnection: List[Tuple[int,int,int,int,int]] = []

    def clear(self):
        self.xSpacing = 0.0
        self.ySpacing = 0.0
        self.entries.clear()
        self.pointsInsideModel.clear()
        self.pointsConnection.clear()

    def fromDesiredNumberOfDivisions(self, heView:HeView, desiredX, desiredY):
        # NOTE - treat the entries
        cloudParams = SimulationParameters()
        if desiredX * desiredY <= 0 or heView.isEmpty():
            self.clear()
            return None
        # NOTE - get the bounding box from the hemodel's view
        xmin, xmax, ymin, ymax = heView.getBoundBox()
        # if not xmin or not xmax or not ymin or not ymax:
        #     raise TypeError(
        #         f'''mywindow.__actionCreatePoinCloud>>> xmin or xmax or ymin or ymax isn't float {xmin=}, {xmax=}, {ymin=}, {ymax=}''')

        # NOTE - divide the model boundbox in x and y subdivisions retrieved from dialog
        xDivisionSize = (xmax - xmin) / (desiredX)
        yDivisionSize = (ymax - ymin) / (desiredY)
        self.xSpacing = xDivisionSize
        self.ySpacing = yDivisionSize

        # SECTION - grid generation
        globalToLocalNthElem = [0 for _ in range( (desiredX+1)*(desiredY+1) )]
        nthElemGlobal = 0
        nthElemLocal = 0
        # NOTE - retrieve model patches
        patches = heView.getPatches()
        tempPoint = Point()
        self.entries.clear()
        self.pointsInsideModel.clear()
        for (i,j) in ((i,j) for j in range(desiredY + 1) for i in range(desiredX + 1)):
            # NOTE - element counting start from 1
            nthElemGlobal = ( j*(desiredX+1)+i ) +1

            tempPoint.setX(xmin + i * xDivisionSize)
            tempPoint.setY(ymin + j * yDivisionSize)

            # NOTE - insert point if inside the model
            for patch in patches:
                if patch.isPointInside(tempPoint):
            # if nthElemGlobal not in [7,8,9,17,18,19]:
                    anEntry = MyCloudPointEntry()
                    anEntry.coord = (tempPoint.getX(), tempPoint.getY())
                    anEntry.position = (i,j)
                    self.entries.append(anEntry)

                    self.pointsInsideModel.append((tempPoint.getX(), tempPoint.getY()))
                    # nthElem += 1
                    nthElemLocal += 1
                    globalToLocalNthElem[nthElemGlobal-1] = nthElemLocal

        # SECTION - generate connection
        fullGridNeighbours = [0,0,0,0]
        cleanedGridNeighbours = [0,0,0,0]
        self.pointsConnection.clear()
        nthElemLocal = 0
        # print(f'{globalToLocalNthElem=}')
        for i in range(len(globalToLocalNthElem)):
            if globalToLocalNthElem[i]:
                self.neighboursOnFullGrid(fullGridNeighbours,desiredX+1,desiredY+1,i+1)
                # print(f'{i+1}: {fullGridNeighbours=}')
                self.neighboursOnCleanedFromFullGrid(cleanedGridNeighbours,fullGridNeighbours,globalToLocalNthElem)
                # print(f'{i+1}: {cleanedGridNeighbours=}')
                self.pointsConnection.append(tuple(cleanedGridNeighbours))
                self.entries[nthElemLocal].neighbors = tuple(cleanedGridNeighbours)
                nthElemLocal += 1
        # print(f'{connect=}')
        # !SECTION

        # NOTE - parameterize the pointCloud
        # cloudParams.particle_half_xSize = abs(xDivisionSize) / 2
        # cloudParams.particle_half_ySize = abs(yDivisionSize) / 2
        # cloudParams.particle_coords = pointCloud
        # cloudParams.particle_restricted = [[0.0]*len(pointCloud[0])]*len(pointCloud)
        # cloudParams.particle_external_force = [[0.0]*len(pointCloud[0])]*len(pointCloud)
        # cloudParams.particle_connection = neighbors

    def neighboursOnFullGrid(self,outVector,x,y,nthElem):
        '''
        considering the couting from right to left, from bottom to top
        `outVector` going to be [rNeighbourId,lNeighbourId,bNeighbourId,tNeighbourId]
        '''
        outVector[:] = [0,0,0,0]
        i = (nthElem-1)%x
        j = (nthElem-1)//x

        if i < x-1:       # NOTE - there is a right neighbour
            outVector[0] = ( j*x+(i+1) ) +1
        if i > 0:         # NOTE - there is a left neighbour
            outVector[1] = ( j*x+(i-1) ) +1
        if j > 0:         # NOTE - there is a bottom neighbour
            outVector[2] = ( (j-1)*x+i ) +1
        if j < y-1:  # NOTE - there is a top neighbour
            outVector[3] = ( (j+1)*x+i ) +1

    def neighboursOnCleanedFromFullGrid(self,outVector,neighboursOnFullGrid,globalToLocalIds):
        outVector[:] = map( lambda globalNthElem: globalToLocalIds[globalNthElem-1] if globalNthElem-1 > -1 else 0, neighboursOnFullGrid)
    # !SECTION
