# coding=UTF-8
################## Детектор маркетов ARUCO
# import the necessary packages

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2.aruco

# import glob

# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]
# print 'mtx=',mtx,'dist=',dist
# exit(0)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)  # (800, 600)#
camera.framerate = 24  # 32
rawCapture = PiRGBArray(camera, size=(640, 480))  # (800, 608)

# allow the camera to warmup
time.sleep(0.1)
font = cv2.FONT_HERSHEY_SIMPLEX
framenumber = 0
markerLength = 13.3  # 133 миллиметра сторона маркера

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    # увеличим яркость, дабы робот не ослеп
    cv2.convertScaleAbs(img, img, 1.8, 0)  # img = img * 2 #convertTo(img,-1,1.5,50)
    # обработаем картинку
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        rvecs = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
        print('corners = ', corners)
        print('rvecs =', rvecs)
        # exit(0)
        i = 0
        while i < len(rvecs[0]):
            cv2.aruco.drawAxis(img, mtx, dist, rvecs[0][i], rvecs[1][i], 5)
            i = i + 1
    # "r: "+str(rvecs[0][0])+
    # cv2.putText(img," v:"+str(rvecs[0][0]),(10,450), font, 1,(0,128,0),1,cv2.LINE_AA)
    cv2.imshow('img', img)
    # cv2.imshow("Frame", gray) #image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    framenumber = framenumber + 1
    # print framenumber, ret
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):  # or framenumber>20:
        break

# cv2.destroyAllWindows()



















import cv2
import numpy as np

class CameraCalibration:
    def __init__(self):
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.objp = np.zeros((6 * 7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def run(self):
        cap = cv2.VideoCapture(0)

        # Создаем окно с флагом cv2.WINDOW_NORMAL
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # Устанавливаем размер окна
        cv2.resizeWindow('img', 1280, 760)

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            img = frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

                # Draw a red square around the chessboard
                corners = corners2.reshape(-1, 2)
                top_left = tuple(corners[0].ravel())
                top_right = tuple(corners[6].ravel())
                bottom_right = tuple(corners[-1].ravel())
                bottom_left = tuple(corners[-7].ravel())
                pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                #calibration
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # undistort
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

                # crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]

                # cv2.imwrite('calibresult.png', dst)
                # cv2.imshow('img', dst)

            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_calibration = CameraCalibration()
    camera_calibration.run()
