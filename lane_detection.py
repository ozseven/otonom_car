import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # Önceki fitler (stabilizasyon için)
        self.left_fit = None
        self.right_fit = None

    def get_birds_eye_view(self, frame):
        height, width = frame.shape[:2]
        src = np.float32([
            [width * 0.45, height * 0.63],
            [width * 0.55, height * 0.63],
            [width * 0.90, height],
            [width * 0.10, height]
        ])
        dst = np.float32([
            [width * 0.2, 0],
            [width * 0.8, 0],
            [width * 0.8, height],
            [width * 0.2, height]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (width, height))
        return warped, M

    def inverse_perspective(self, image, M):
        height, width = image.shape[:2]
        Minv = np.linalg.inv(M)
        return cv2.warpPerspective(image, Minv, (width, height))

    def apply_thresholds(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        _, binary = cv2.threshold(scaled_sobel, 50, 255, cv2.THRESH_BINARY)
        return binary

    def find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        margin = 100
        minpix = 50

        window_height = np.int32(binary_warped.shape[0] // nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped, leftx, lefty, rightx, righty):
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            self.left_fit = left_fit
            self.right_fit = right_fit
        except:
            left_fit = self.left_fit
            right_fit = self.right_fit
        return left_fit, right_fit

    def draw_lanes(self, original_img, binary_warped, left_fit, right_fit, M):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, [np.int32(pts)], (0, 255, 0))
        newwarp = self.inverse_perspective(color_warp, M)
        result = cv2.addWeighted(original_img, 1, newwarp, 0.4, 0)
        return result

    def detect_lanes(self, frame):
        warped, M = self.get_birds_eye_view(frame)
        binary = self.apply_thresholds(warped)
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary)
        left_fit, right_fit = self.fit_polynomial(binary, leftx, lefty, rightx, righty)
        result = self.draw_lanes(frame, binary, left_fit, right_fit, M)
        return result