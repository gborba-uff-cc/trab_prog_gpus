import sys
import os
import configparser

from mywindow import *

def main():
    app = QApplication(sys.argv)
    gui = MyWindow()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
