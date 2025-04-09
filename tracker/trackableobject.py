import re
import math
class TrackableObject:
	def __init__(self, objectID, centroid, name, bbox, state, time, status, stillness_threhold, movingCallback=None, roiPencurianMotor=None):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = centroid
		self.name = name
		self.bbox = bbox
		self.state = state
		self.stillness_threshold = stillness_threhold
		self.isStill = False
		self.movingCallback = movingCallback
		self.frame = None
		self.roiPencurianMotor = roiPencurianMotor
		self.isSendPencurianMotor = False
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		# self.self_count = self_count
		self.time = time
		self.status = status

	def setCentroid(self,centroid):
		if len(self.centroids) < 2:
			self.centroids.append(centroid)
			return
		centroid2 = self.centroids[-2]
		isMoving = math.sqrt((centroid[0] - centroid2[0])**2 + (centroid[1] - centroid2[1])**2) > self.stillness_threshold
		if self.isStill and isMoving and self.movingCallback is not None and self.name == "Motor":
			self.movingCallback(self, centroid)
		self.isStill = not isMoving
		self.centroids.append(centroid)

	def setPlate(self,plates):
		pattern = r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$'
		plates = filter(lambda x: re.match(pattern, x[0]), plates)
		for plate in plates:
			if len(self.plate) < len(plate):
				self.plate = plate