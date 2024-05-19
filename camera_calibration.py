import cv2
import numpy as np


class CameraCalibration:
    def __init__(self):
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.objp = np.zeros((6 * 7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def drawSquare(self, img, corners2):
        corners = corners2.reshape(-1, 2)
        top_left = tuple(corners[0].ravel())
        top_right = tuple(corners[6].ravel())
        bottom_right = tuple(corners[-1].ravel())
        bottom_left = tuple(corners[-7].ravel())
        pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        return img

    def run(self):
        cap = cv2.VideoCapture(0)

        # Создаем окно с флагом cv2.WINDOW_NORMAL
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # Устанавливаем размер окна
        cv2.resizeWindow('img', 1280, 760)

        count = 0
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
                # self.drawSquare()

                # Calibration
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],None, None)
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # Undistort
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

                # Crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]

                if count < 10:
                    gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                    ret_dst, corners_dst = cv2.findChessboardCorners(gray_dst, (7, 6), None)
                    if ret_dst:
                        corners2_dst = cv2.cornerSubPix(gray_dst, corners_dst, (11, 11), (-1, -1), self.criteria)
                        dst_with_square = self.drawSquare(dst.copy(), corners2_dst)
                        img_with_square = self.drawSquare(img.copy(), corners2)
                        is_saved = cv2.imwrite(f"files/calibrated_{count}.png", dst_with_square)
                        is_saved_img = cv2.imwrite(f"files/original_{count}.png", img_with_square)
                        if is_saved and is_saved_img:
                            print(f"photo saved: {count}")
                            count += 1

                # # Показ откалиброванного изображения
                cv2.imshow('img', dst)
            else:
                cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_calibration = CameraCalibration()
    camera_calibration.run()
