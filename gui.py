import cv2
import numpy as np

import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

from PySide6 import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QThread, Signal, Slot, Qt

import imutils
from dvs import DvsCam

class CameraThread(QThread):
  frame_signal = QtCore.Signal(QImage)
  dvsframe_signal = QtCore.Signal(QImage)
  event_signal = QtCore.Signal(np.ndarray)

  def run(self):
    self.cap = DvsCam().pair(cv2.VideoCapture(0))
    while self.cap.isOpened():
      frame, dvsFrame, dvsEvents = self.cap.read()
      frame, dvsFrame = self.cvimage_to_label(frame), self.cvimage_to_label(dvsFrame)
      self.frame_signal.emit(frame)
      self.dvsframe_signal.emit(dvsFrame)
      self.event_signal.emit(dvsEvents)

  def cvimage_to_label(self,image):
    image = np.array(image, np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = QImage(image,
                   image.shape[1],
                   image.shape[0],
                   QImage.Format_RGB888)
    return image


class MainApp(QMainWindow):
  def __init__(self):
    super().__init__()
    self.index = 0
    self.event_array = None
    self.recording = False
    self.init_ui()
    self.show()

  def init_ui(self):
    # Setup Window
    self.showMaximized()
    self.setWindowTitle("CVPR Asl Demo")

    # Setup Layout
    page_layout = QtWidgets.QVBoxLayout()
    image_layout = QtWidgets.QHBoxLayout() 
    button_layout = QtWidgets.QHBoxLayout()

    # Image labels
    self.frame_label = QtWidgets.QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
    image_layout.addWidget(self.frame_label)

    self.dvs_frame_label = QtWidgets.QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
    image_layout.addWidget(self.dvs_frame_label)
    
    page_layout.addLayout(image_layout)

    # Class Label
    self.class_label = QtWidgets.QLabel(str(self.index))
    self.class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    page_layout.addWidget(self.class_label)

    # Buttons
    self.record_button = QtWidgets.QPushButton("Record")
    self.record_button.pressed.connect(self.start_recording)
    button_layout.addWidget(self.record_button)

    self.predict_button = QtWidgets.QPushButton("Predict")
    self.predict_button.pressed.connect(self.predict)
    button_layout.addWidget(self.predict_button)

    page_layout.addLayout(button_layout)

    # Camera Thread
    self.camera_thread = CameraThread()
    self.camera_thread.frame_signal.connect(self.setFrame)
    self.camera_thread.dvsframe_signal.connect(self.setDvsFrame)
    self.camera_thread.event_signal.connect(self.storeEvents)
    self.camera_thread.start()

    # Setup Widget
    widget = QWidget(self)
    widget.setLayout(page_layout)
    self.setCentralWidget(widget)

  def predict(self):
    pass

  def start_recording(self):
    self.recording = True
    self.event_array = None
    self.index = 0

  @QtCore.Slot(np.ndarray)
  def storeEvents(self, events):
    if self.event_array is None:
      self.event_array = np.zeros(shape=(90, 2, events.shape[1], events.shape[2]))

    if self.recording:
      self.event_array[self.index] = events
      self.index = (self.index + 1)
      self.class_label.setText(str(self.index))

      if self.index == 90:
        self.recording = False


  @QtCore.Slot(QImage)
  def setFrame(self, image):
    self.frame_label.setPixmap(QPixmap.fromImage(image))

  
  @QtCore.Slot(QImage)
  def setDvsFrame(self, image):
    self.dvs_frame_label.setPixmap(QPixmap.fromImage(image))