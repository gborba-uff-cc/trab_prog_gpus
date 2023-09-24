from typing import List, Tuple, Union


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

        xmin, xmax = (None, ) * 2
        ymin, ymax = (None, ) * 2

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
