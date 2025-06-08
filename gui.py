import cv2
import numpy as np

import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

# from PySide6 import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import QThread, Signal, Slot, Qt

import imutils
from dvs import DvsCam
from infer import predict_sample

class CameraThread(QThread):
  frame_signal = QtCore.Signal(QImage)
  dvsframe_signal = QtCore.Signal(QImage)
  event_signal = QtCore.Signal(np.ndarray)
  data_signal = QtCore.Signal(np.ndarray)

  def run(self):
    self.cap = DvsCam().pair(cv2.VideoCapture(0))
    while self.cap.isOpened():
      frame, dvsFrame, dvsEvents = self.cap.read()
      self.data_signal.emit(frame)
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
    self.countdown = False
    self.init_ui()
    self.show()

  def init_ui(self):
    # Setup Window
    self.showMaximized()
    self.setWindowTitle("CVPR Asl Demo")

    # Setup Layout
    page_layout = QtWidgets.QVBoxLayout()
    image_layout = QtWidgets.QHBoxLayout() 
    label_layout = QtWidgets.QVBoxLayout() 
    button_layout = QtWidgets.QHBoxLayout()
    data_layout = QtWidgets.QVBoxLayout()

    # Image labels
    self.frame_label = QtWidgets.QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
    image_layout.addWidget(self.frame_label)

    self.dvs_frame_label = QtWidgets.QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
    image_layout.addWidget(self.dvs_frame_label)
    
    page_layout.addLayout(image_layout)

    # Class Label
    self.display_label = QtWidgets.QLabel("Ready to Record\n (Press Record)", alignment=Qt.AlignmentFlag.AlignCenter)
    self.display_label.setFont(QFont("Times", 24))
    label_layout.addWidget(self.display_label)

    self.class_label = QtWidgets.QLabel("Class: _______", alignment=Qt.AlignmentFlag.AlignCenter)
    self.class_label.setFont(QFont("Times", 20))
    label_layout.addWidget(self.class_label)

    page_layout.addLayout(label_layout)

    # Buttons
    self.record_button = QtWidgets.QPushButton("Record")
    self.record_button.pressed.connect(self.start_recording)
    button_layout.addWidget(self.record_button)

    self.predict_button = QtWidgets.QPushButton("Predict")
    self.predict_button.pressed.connect(self.predict)
    button_layout.addWidget(self.predict_button)

    page_layout.addLayout(button_layout)

    
    # Bandwidth
    data_label_font = QFont("Times", 18)

    bandwidth_layout = QtWidgets.QHBoxLayout()

    self.rgb_bw_label = QtWidgets.QLabel("000 bits", alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, font=data_label_font)
    bw_label = QtWidgets.QLabel("Bandwidth", alignment=Qt.AlignmentFlag.AlignCenter, font=data_label_font)
    self.dvs_bw_label = QtWidgets.QLabel("000 bits", alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, font=data_label_font)

    bandwidth_layout.addWidget(self.rgb_bw_label)
    bandwidth_layout.addWidget(bw_label)
    bandwidth_layout.addWidget(self.dvs_bw_label)

    data_layout.addLayout(bandwidth_layout)

    # Throughput
    throughput_layout = QtWidgets.QHBoxLayout()

    self.rgb_tp_label = QtWidgets.QLabel("000 bits", alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, font=data_label_font)
    tp_label = QtWidgets.QLabel("Throughput", alignment=Qt.AlignmentFlag.AlignCenter, font=data_label_font)
    self.dvs_tp_label = QtWidgets.QLabel("000 bits", alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, font=data_label_font)

    throughput_layout.addWidget(self.rgb_tp_label)
    throughput_layout.addWidget(tp_label)
    throughput_layout.addWidget(self.dvs_tp_label)

    data_layout.addLayout(throughput_layout)

    # Data Transfer
    datatransfer_layout = QtWidgets.QHBoxLayout()

    self.rgb_dt_label = QtWidgets.QLabel("000 bits", alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, font=data_label_font)
    dt_label = QtWidgets.QLabel("Data Transfer", alignment=Qt.AlignmentFlag.AlignCenter, font=data_label_font)
    self.dvs_dt_label = QtWidgets.QLabel("000 bits", alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, font=data_label_font)

    datatransfer_layout.addWidget(self.rgb_dt_label)
    datatransfer_layout.addWidget(dt_label)
    datatransfer_layout.addWidget(self.dvs_dt_label)
    
    data_layout.addLayout(datatransfer_layout)


    page_layout.addLayout(data_layout)


    # Camera Thread
    self.camera_thread = CameraThread()
    self.camera_thread.frame_signal.connect(self.setFrame)
    self.camera_thread.dvsframe_signal.connect(self.setDvsFrame)
    self.camera_thread.event_signal.connect(self.storeEvents)
    self.camera_thread.data_signal.connect(self.frameStats)
    self.camera_thread.start()

    # Setup Widget
    widget = QWidget(self)
    widget.setLayout(page_layout)
    self.setCentralWidget(widget)

  def predict(self):
    if self.index == 90:
      pred = predict_sample(self.event_array)
      self.class_label.setText(f'Class: {pred}')
      self.display_label.setText("Ready to Record\n (Press Record)")


  def start_recording(self):
    self.countdown = True
    self.event_array = None
    self.counter = 90

  @QtCore.Slot(np.ndarray)
  def storeEvents(self, events):
    if self.event_array is None:
      self.event_array = np.zeros(shape=(90, 2, events.shape[1], events.shape[2]))

    if self.countdown:
      if self.counter <= 0:
        self.recording = True
        self.index = 0
        self.countdown = False
      else:
        self.counter -= 1
        label = f'Getting Ready \n{1 + (self.counter // 30)}...'
        self.display_label.setText(label)


    elif self.recording:
      self.event_array[self.index] = events
      self.index = (self.index + 1)
      label = f'Recording \n {progressBar(self.index, max=90, length = 20)}'
      self.display_label.setText(label)

      if self.index == 90:
        self.recording = False
        self.display_label.setText("Ready to Predict! \n (Press Predict)")

    # Bandwidth
    data_bw_bits = np.prod(events.shape) * 1 * 30
    self.dvs_bw_label.setText(f'{formatBits(data_bw_bits)}bits/s')

    
    # Throughput
    data_tp_bits = np.count_nonzero(events) * 30
    self.dvs_tp_label.setText(f'{formatBits(data_tp_bits)}events/s')



  def frameStats(self, frame):

    # Bandwidth
    frame_bits = np.prod(frame.shape) * 8 * 30
    self.rgb_bw_label.setText(f'{formatBits(frame_bits)}bits/s')

  @QtCore.Slot(QImage)
  def setFrame(self, image):
    self.frame_label.setPixmap(QPixmap.fromImage(image))


  
  @QtCore.Slot(QImage)
  def setDvsFrame(self, image):
    self.dvs_frame_label.setPixmap(QPixmap.fromImage(image))




def formatBits(bits):
  endSet = ["", "kilo", "mega", "giga"]

  end = 0

  while bits >= 1024:
    bits = bits / 1024
    end += 1

  return f'{round(bits,1)} {endSet[end]}'

def progressBar(i, max=100, length=10):
  
  filledSymbol = "■"
  emptySymbol = "□"
  progress = int((i / max) * length)
  bar = ""
  for _ in range(progress): 
    bar += filledSymbol
  for _ in range(length-progress):
    bar += emptySymbol
  return bar