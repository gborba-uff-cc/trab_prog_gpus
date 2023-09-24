from typing import List, Tuple
from OpenGL.GL import *
from PyQt5 import QtOpenGL, QtCore
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import *

from hetool import *

from mymodel import MyModel  # import para typing


class MyCanvas(QtOpenGL.QGLWidget):
    def __init__(self):
        super().__init__()
        self.__model = None

        # NOTE - hetool MVC
        # NOTE - will be reinjected from mywindow
        self.hemodel: HeModel = HeModel()
        self.heview: HeView = HeView(self.hemodel)
        self.hecontroller: HeController = HeController(self.hemodel)
        self.hetol = 1e-6

        # NOTE - cloud with points inside the model
        self.hepointCloud: List[Tuple[float, float]] = []

        self.__w = 0  # width: GL canvas horizontal size
        self.__h = 0  # height: GL canvas vertical size

        self.__L: float = -1000.0
        self.__R: float =  1000.0
        self.__B: float = -1000.0
        self.__T: float =  1000.0

        self.__list = None  # the methods involving list are because a bug messing thing up when erasing the buffers

        self.__buttonPressed = False
        self.__mouseDragged = False
        self.__pt0 = QtCore.QPoint(0, 0)
        self.__pt1 = QtCore.QPoint(0, 0)

    def initializeGL(self):
        # glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_LINE_SMOOTH)
        self.__list = glGenLists(1)

    def resizeGL(self, width, height):
        # store GL canvas sizes in object properties
        self.__w = width
        self.__h = height

        if self.__model is None or self.__model.isEmpty():
            self.scaleWorldWindow(1.0)
        else:
            self.__L, self.__R, self.__B, self.__T = self.__model.getBoundBox()
            # adding margins to the model
            self.scaleWorldWindow(1.1)

        # setup the viewport to canvas dimensions
        glViewport(0, 0, self.__w, self.__h)

        self.setOrthographicProjectionClippingVolume()

        # setup display in model coordinates
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        # clear the buffer with the current clear color
        glClear(GL_COLOR_BUFFER_BIT)

        glCallList(self.__list)
        glDeleteLists(self.__list, 1)
        self.__list = glGenLists(1)
        glNewList(self.__list, GL_COMPILE)

        # NOTE - canvas grid
        glColor3f(0.5, 0.5, 0.5)
        glPointSize(2)
        glBegin(GL_POINTS)
        xi = int(self.__L)
        while xi <= int(self.__R):
            yi = int(self.__B)
            while yi <= int(self.__T):
                glVertex2d(xi,yi)
                yi += 100
            xi += 100
        glEnd()

        # NOTE - Desenho dos pontos coletados
        pt0_U = self.convertPtCoordsToUniverse(self.__pt0)
        pt1_U = self.convertPtCoordsToUniverse(self.__pt1)
        glColor3f(0.0, 1.0, 0.0) # green
        glBegin(GL_LINE_STRIP)
        glVertex2f(pt0_U.x(), pt0_U.y())
        glVertex2f(pt1_U.x(), pt1_U.y())
        glEnd()

        # # NOTE - drawing model
        # if self.__model is not None and not self.__model.isEmpty():
        #     verts = self.__model.verts
        #     # glColor3f(0.0, 1.0, 0.0)  # green
        #     # glBegin(GL_TRIANGLES)
        #     # for vtx in verts:
        #     #     glVertex2f(vtx.x, vtx.y)
        #     # glEnd()
        #     curves = self.__model.curves
        #     glColor3f(0.0, 0.0, 1.0)  # blue
        #     glBegin(GL_LINES)
        #     for curv in curves:
        #         glVertex2f(curv.p1.x, curv.p1.y)
        #         glVertex2f(curv.p2.x, curv.p2.y)
        #     glEnd()

        # NOTE - drawing hemodel
        if not(self.heview.isEmpty()):
            # print("teste")
            patches = self.heview.getPatches()
            # STUB - print
            # print(f'{len(patches)=}')
            glColor3f(1.0, 0.0, 1.0) #
            glBegin(GL_TRIANGLES)
            for pat in patches:
                triangs = Tesselation.tessellate(pat.getPoints())
                for triang in triangs:
                    for pt in triang:
                        glVertex2d(pt.getX(),pt.getY())
            glEnd()

            segments = self.heview.getSegments()
            # STUB - print
            # print(f'{len(segments)=}')
            glColor3f(0.0, 1.0, 1.0)
            glBegin(GL_LINES)
            for curv in segments:
                ptc = curv.getPointsToDraw()
                # for pt in ptc:
                #     print(pt.getX(), pt.getY())
                glVertex2f(ptc[0].getX(), ptc[0].getY())
                glVertex2f(ptc[1].getX(), ptc[1].getY())
            glEnd()

            verts = self.heview.getPoints()
            glColor3f(1.0, 1.0, 0.0)
            glPointSize(5)
            glBegin(GL_POINTS)
            for vert in verts:
                glVertex2f(vert.getX(),vert.getY())
            glEnd()

            glColor3f(1.0, 0.8, 0.8)
            glPointSize(5)
            glBegin(GL_POINTS)
            for x, y in self.hepointCloud:
                glVertex2f(x, y)
            glEnd()

        glEndList()

    def setCanvasGrid(self):
        pass

    def setModel(self, model: MyModel):
        self.__model = model

    def fitWorldToViewport(self):
        """
        Faz que o mundo caiba no viewport sem distorcer os elementos.
        """
        # if self.__model is None or self.heview.isEmpty():
        #     return
        # self.__L, self.__R, self.__B, self.__T = self.__model.getBoundBox()

        if self.heview.isEmpty():
            return
        self.__L, self.__R, self.__B, self.__T = self.heview.getBoundBox()

        self.scaleWorldWindow(1.10)

        self.update()

    def scaleWorldWindow(self, scale_factor):
        # compute canvas viewport distortion ratio
        vpr = self.__w / self.__h

        # get the current window (world?) center
        cx = (self.__L + self.__R) * 0.5
        cy = (self.__B + self.__T) * 0.5

        # set new window (world?) sizes based on scaling factor
        sizex = (self.__R - self.__L) * scale_factor
        sizey = (self.__T - self.__B) * scale_factor

        # adjust window (world?) to keep the same aspect ratio of the viewport.
        if sizex > sizey * vpr:
            sizey = sizex / vpr
        else:
            sizex = sizey * vpr
        self.__L = cx - (sizex * 0.5)
        self.__R = cx + (sizex * 0.5)
        self.__B = cy - (sizey * 0.5)
        self.__T = cy + (sizey * 0.5)

        self.setOrthographicProjectionClippingVolume()

    def panWorldWindow(self, pan_factor_x, pan_factor_y):
        # compute pan distances in horizontal and vertical directions
        panx = (self.__R - self.__L) * pan_factor_x
        pany = (self.__B - self.__T) * pan_factor_y

        # shift current window
        self.__L += panx
        self.__R += panx
        self.__B += pany
        self.__T += pany

        self.setOrthographicProjectionClippingVolume()
        self.update()

    def setOrthographicProjectionClippingVolume(self):
        """
        establish the clipping volume by setting up an ortographic projection
        """
        # reset the coordinate system
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # setando a area visivel mundo
        glOrtho(self.__L, self.__R, self.__B, self.__T, -1.0, 1.0)

    def convertPtCoordsToUniverse(self, point):
        dX = self.__R - self.__L
        dY = self.__T - self.__B
        mX = point.x() * dX / self.__w
        mY = (self.__h - point.y()) * dY / self.__h
        x = self.__L + mX
        y = self.__B + mY
        return QtCore.QPointF(x, y)

    def mousePressEvent(self, event: QMouseEvent):
        self.__buttonPressed = True
        self.__pt0 = event.pos()
#       print(f'mousePressEvent: pt0 event: {event.pos().x()},{event.pos().y()}')

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.__buttonPressed:
            self.__mouseDragged = True
            self.__pt1 = event.pos()
#           print(f'mouseMoveEvent: pt1 event: {event.pos().x()},{event.pos().y()}')
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        pt0_U = self.convertPtCoordsToUniverse(self.__pt0)
        pt1_U = self.convertPtCoordsToUniverse(self.__pt1)

        x0seg, y0seg, x1seg, y1seg = (0.0, 0.0, 0.0, 0.0)
        snap_tol = 50
        snapped0, x0snapped, y0snapped = self.heview.snapToPoint(pt0_U.x(), pt0_U.y(), snap_tol)
        snapped1, x1snapped, y1snapped = self.heview.snapToPoint(pt1_U.x(), pt1_U.y(), snap_tol)

        if snapped0:
            x0seg = x0snapped
            y0seg = y0snapped
        else:
            x0seg = pt0_U.x()
            y0seg = pt0_U.y()

        if snapped1:
            x1seg = x1snapped
            y1seg = y1snapped
        else:
            x1seg = pt1_U.x()
            y1seg = pt1_U.y()

        if self.__mouseDragged and (pt0_U.x() != pt1_U.x() or pt0_U.y() != pt1_U.y()):
            # self.__model.addCurve(x0seg, y0seg, x1seg, y1seg)
            self.hecontroller.insertSegment([x0seg, y0seg, x1seg, y1seg], self.hetol)
            self.update()
            self.paintGL()
#            print(f'mouseReleaseEvent: added pt0:{self.__pt0.x()},{self.__pt0.y()} pt1:{self.__pt1.x()},{self.__pt1.y()}')
        self.__buttonPressed = False
        self.__mouseDragged = False
        self.__pt0.setX(0)
        self.__pt0.setY(0)
        self.__pt1.setX(0)
        self.__pt1.setY(0)
