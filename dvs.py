import numpy as np

class DvsCam():
  def __init__(self, intensityThreshold=0.05, onEventColor=(162, 249, 84), offEventColor=(255, 139, 237)):
    self.camera = None
    self.previousFrame = None
    self.intensityThreshold = intensityThreshold
    self.onEventColor = onEventColor
    self.offEventColor = offEventColor
  
  def isOpened(self):
    if self.camera is None: return False
    else: return self.camera.isOpened()
  def imageWidth(self):
    if self.previousFrame is None: return None
    return self.previousFrame.shape[0]
  def imageHeight(self):
    if self.previousFrame is None: return None
    return self.previousFrame.shape[1]
  

  def pair(self, camera):

    ret, frame = camera.read()
    
    if ret is False: 
      print(f'{camera} could not be opened')
      return self
    
    self.camera = camera
    self.previousFrame = self._prepImage(frame)
    return self

  def read(self):
    if self.camera is None: return None
    
    # Read Camera
    _, frame = self.camera.read()
    frame = self._prepImage(frame)

    # Convert to DVS
    dvsColorized, events = self._dvs(frame)

    # Store current frame as previous frame
    self.previousFrame = frame

    return frame * 255., dvsColorized * 255., events

  def _prepImage(self, img): return np.array(img, np.float32) / 255.
  def _dvs(self, frame):
  
    # Find difference of images
    diff = np.mean(frame - self.previousFrame, axis=-1)

    # Apply Thresholding
    diff = np.where( np.abs(diff) < self.intensityThreshold, 0, diff)
    
    # Add Color
    ret = np.zeros_like(self.previousFrame)
    ret[diff > 0, :] = self.onEventColor
    ret[diff < 0, :] = self.offEventColor

    events = np.zeros(shape=(2, ret.shape[0], ret.shape[1]))
    events[0, diff > 0] = 1
    events[1, diff < 0] = 1

    return ret, events
