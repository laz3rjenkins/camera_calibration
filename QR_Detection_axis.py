import cv2
import cv2.aruco
import numpy as np
from pyzbar.pyzbar import decode

class QRCodeDetector:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.markerLength = 13.3
        # with np.load('B.npz') as X:
        #     self.mtx, self.dist, _, _ = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

    def detect_qr_codes(self, frame):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find QR codes in the frame
        qr_codes = decode(gray_frame)

        return qr_codes

    def extract_qr_data(self, qr_codes):
        if qr_codes:
            for qr_code in qr_codes:
                # Extract the data from the QR code
                qr_data = qr_code.data.decode('utf-8')
                print("QR Code Data:", qr_data)

    def detect_codes(self, frame):
        # # Convert the frame to grayscale
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # # Find QR codes in the frame
        # qr_codes = decode(gray_frame)

        # img = frame.array
        img = frame
        print(frame)
        # images_gray = np.zeros(images.shape[:-1], dtype=images.dtype)
        # for i, img in enumerate(images):
        #     images_gray[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = images_gray
        # увеличим яркость, дабы робот не ослеп
        cv2.convertScaleAbs(img, img, 1.8, 0)
        # обработаем картинку
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.ArucoDetector.detectMarkers(img, gray)

        return {
            'corners': corners,
            'ids': ids,
            'rejected': rejected,
            'img': img
        }

    def draw_axis(self, codes):
        corners = codes['corners']
        ids = codes['ids']
        rejected = codes['rejected']
        img = codes['img']
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            rvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.mtx, self.dist)
            print('corners = ', corners)
            print('rvecs =', rvecs)
            # exit(0)
            i = 0
            while i < len(rvecs[0]):
                cv2.aruco.drawAxis(img, self.mtx, self.dist, rvecs[0][i], rvecs[1][i], 5)
                i = i + 1


    def draw_qr_code_rectangles(self, frame, qr_codes):
        if qr_codes:
            for qr_code in qr_codes:
                # Draw a rectangle around the QR code on the frame
                points = qr_code.polygon
                print(points)
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    cv2.polylines(frame, [hull], True, (255, 0, 255), 3)
                else:
                    cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (255, 0, 255), 3)

    def run(self):
        cap = cv2.VideoCapture(0)
        # camera = PiCamera()
        # camera.resolution = (640, 480)  # (800, 600)#
        # camera.framerate = 24  # 32

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            # qr_codes = self.detect_qr_codes(frame)
            qr_codes = self.detect_codes(frame)
            self.draw_axis(qr_codes)
            # self.extract_qr_data(qr_codes)
            # self.draw_qr_code_rectangles(frame, qr_codes)

            # Display the frame with detected QR codes
            cv2.imshow("Detect QR Code from Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    qr_detector = QRCodeDetector()
    qr_detector.run()
