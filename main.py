import cv2
from dvs import DvsCam

import sys
from PySide6 import QtWidgets
from gui import MainApp

def main():

  # Open Qt Application
  app = QtWidgets.QApplication([])

  # Open Camera
  # cam = DvsCam().pair(cv2.VideoCapture(0))


  # frame, dvsFrame = cam.read(returnOriginal=True)


  main_window = MainApp()

  # Close Application
  sys.exit(app.exec())


main()