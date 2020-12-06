import numpy as np
import cv2

def preprocess_image(img):
    '''
    Gray Scale
    Gaussian Blur
    Sobel X
    Canny
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=1)
    sobel_x = np.uint8(np.absolute(sobel_x))
    canny_x = cv2.Canny(sobel_x, 40, 150)
    canny = cv2.Canny(blur, 40, 150)
    return gray, blur, sobel_x, canny_x, canny

def make_roi(frame):
    height, width = frame.shape[:2]
    mask = np.zeros(frame.shape, np.uint8)
    points = np.array([
        [width * (0.5 - 0.15), height * (0.33)],
        [width * (0.5 + 0.15), height * (0.33)],
        [(width) * (0.5 + 0.2), height],
        [(width) * (0.5 - 0.2), height],
    ], np.int32)
    points = points.reshape((-1, 1, 2))
    mask_line = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    mask = cv2.fillPoly(mask_line.copy(), [points], (255, 255, 255))

    ROI = cv2.bitwise_and(mask, frame)
    return ROI
    
def warp_bird_eye_view(img):
    pass

def detect_rail_lane(img, previmg=None):
    pass

def sift_move_line(image2, canny_x_ROI, gray):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=True)

    kp_old, des_old = sift.detectAndCompute(canny_x_ROI, None)

    _, _, _, d, _ = preprocess_image(image2)
    d_ROI = make_roi(d)
    kp, des = sift.detectAndCompute(d_ROI, None)

    matches = bf.match(des_old, des)
    matches = sorted(matches, key=lambda x: x.distance)

    drawn = cv2.drawKeypoints(canny_x_ROI, kp_old, canny_x_ROI)
    hi = cv2.drawKeypoints(d_ROI, kp, d_ROI)
    '''
    cv2.imshow("Key Points", drawn)
    cv2.imshow("Key Points 2", hi)
    '''

    wow = cv2.drawKeypoints(gray, kp, gray)
    for match in matches:
        q = match.queryIdx
        t = match.trainIdx
        pp = np.array(kp_old[q].pt).astype(np.int)
        qq = np.array(kp[t].pt).astype(np.int)
        color = (0, 0, 255)
        cv2.line(wow, tuple(pp), tuple(qq), color)
        print(pp, qq)
    return wow

def track_optical_flow(gray, image2):
    p0 = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)

    gray_cur, _, _, _, _ = preprocess_image(image2)

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray, gray_cur, p0, None)#, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    mask = np.copy(image2)
    for new, old in zip(good_new, good_old):
        a, b = new.ravel().astype(np.int)
        c, d = old.ravel().astype(np.int)
        mask = cv2.line(mask, (a,b), (c,d), (0, 0, 255))
    return mask

def dense_flow(gray, gray_cur, image2):
    flow = cv2.calcOpticalFlowFarneback(gray, gray_cur, None, 0.5, 4, 7, 4, 5, 1.1, 0)
    real_mask = np.full_like(gray, 255)
    real_mask = make_roi(real_mask)

    a = np.hypot(flow[:,:,0], flow[:,:,1])
    a[real_mask==0] = 2147483647
    s = a.argsort(axis=None)[:100]

    mask = np.copy(make_roi(image2))
    height, width = gray.shape
    for y in range(2, height, 7):
        for x in range(2, width, 7):
            fx, fy = flow[y,x]
            mask = cv2.line(mask, (x, y), (int(x + fx), int(y + fy)), (0, 0, 255))

    for ss in s:
        yy, xx = np.unravel_index(ss, a.shape)
        cv2.circle(mask, (xx, yy), 1, (255, 0, 0))
    return mask

if __name__ == "__main__":
    image = cv2.imread('./data/images/incheon-magnetic-right4f.png')

    gray, blur, sobel_x, canny_x, canny = preprocess_image(image)
    canny_x_ROI = make_roi(canny_x)

    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blur)
    cv2.imshow("Sobel X", sobel_x)
    cv2.imshow("Canny X", canny_x)
    cv2.imshow("Canny", canny)
    cv2.imshow("Canny X ROI", canny_x_ROI)

    cv2.waitKey(0); cv2.destroyAllWindows()

    image2 = cv2.imread('./data/images/incheon-magnetic-right4l.png')
    '''
    wow = sift_move_line(image2, canny_x_ROI, gray)
    cv2.imshow("Move", wow)
    '''

    '''
    mask = track_optical_flow(gray, image2)
    cv2.imshow("Optical Flow", mask)
    '''

    gray_cur = preprocess_image(image2)[0]

    mask = dense_flow(gray, gray_cur, image2)

    cv2.imshow("OpticalFlow", mask)

    cv2.waitKey(0); cv2.destroyAllWindows()