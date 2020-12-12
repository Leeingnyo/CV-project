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

def make_roi_with(frame, points):
    height, width = frame.shape[:2]
    mask = np.zeros(frame.shape, np.uint8)
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    mask_line = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    mask = cv2.fillPoly(mask_line.copy(), [points], (255, 255, 255))

    ROI = cv2.bitwise_and(mask, frame)
    return ROI
    
def warp_bird_eye_view(img):
    src = np.float32([
        (894, 597), (1010, 598),
        (731, 989), (1197, 989),
    ])
    # below for RCT
    '''
        (890, 690), (1040, 690),
        (730, 1020), (1200, 1020),
    '''
    RATIO = 3
    width = 1920
    height = 1080 * 2
    dst = np.float32([
        (width / 2 - 30 * RATIO, height - 30 * 7 * RATIO), (width / 2 + 30 * RATIO, height - 30 * 7 * RATIO),
        (width / 2 - 30 * RATIO, height - 30 * 1 * RATIO), (width / 2 + 30 * RATIO, height - 30 * 1 * RATIO),
    ])

    transformMatrix = cv2.getPerspectiveTransform(src, dst)

    warped = np.zeros((height, width))
    warped = cv2.warpPerspective(img, transformMatrix, (width, height))

    warped_resize = cv2.resize(warped, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

    return warped_resize

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
        # print(pp, qq)
    return wow

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

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
    return mask, good_new, good_old

def dense_flow(gray, gray_cur, image2):
    flow = cv2.calcOpticalFlowFarneback(gray, gray_cur, None, 0.5, 4, 7, 4, 5, 1.1, 0)
    real_mask = np.full_like(gray, 255)
    real_mask = make_roi(real_mask)

    a = np.hypot(flow[:,:,0], flow[:,:,1])
    a[real_mask==0] = 2147483647
    s = a.argsort(axis=None)[:100]

    mask = np.copy(image2)
    height, width = gray.shape
    for y in range(2, height, 7):
        for x in range(2, width, 7):
            fx, fy = flow[y,x]
            mask = cv2.line(mask, (x, y), (int(x + fx), int(y + fy)), (0, 0, 255))

    for ss in s:
        yy, xx = np.unravel_index(ss, a.shape)
        cv2.circle(mask, (xx, yy), 1, (255, 0, 0))
    return mask, flow

def houghlines(canny):
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 40)
    if lines is not None:
        return [line[0] for line in lines
            if np.pi / 3 >= line[0][1] or line[0][1] >= np.pi * 2 / 3]
    return None

def nearest_lines(lines, expected_lower_x, expected_lower_y):
    min_value = 1920
    min_line = None
    min_lower_x = None
    min_upper_x = None
    for line in lines:
        rho, theta = line
        # rho = x cos theta + y sin theta
        lower_x = (rho - expected_lower_y * np.sin(theta)) / (np.cos(theta) + 1e-5)
        upper_x = (rho) / (np.cos(theta) + 1e-5)
        # print(lower_x)
        value = np.abs(lower_x - expected_lower_x)
        if value < min_value:
            min_line = line
            min_value = value
            min_lower_x = lower_x
            min_upper_x = upper_x
    return min_line, min_lower_x, min_upper_x

if __name__ == "__main__":
    image = cv2.imread('./data/images/incheon-magnetic-1080p-left.png')

    video_name = 'incheon-magnetic-1080p.mp4'
    # video_name = 'RCT3.mp4'
    video_dir = 'data/'
    video_path = video_dir + video_name
    capture = cv2.VideoCapture(video_path) # start from 2880

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    # print(fps, width, height, total_frame)

    # commented out
    # seek = 428 # stop
    # seek = 2880 # stop
    # seek = 4000 # straight
    # seek = 14744 # right
    # seek = 24775 # left
    seek = 28875 # left
    capture.set(cv2.CAP_PROP_POS_FRAMES, seek)

    old_patch_roi = None

    while capture.get(cv2.CAP_PROP_POS_FRAMES) < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, image = capture.read()

        # predefined
        X_LEFT = 700
        X_RIGHT = 1200
        PATCH_H = 80

        height, width = image.shape[:2]

        left_lower_x_ll = []
        left_upper_x_ll = []
        right_lower_x_ll = []
        right_upper_x_ll = []

        lower_y_ll = []
        upper_y_ll = []

        loop = 0
        mask = np.copy(image)
        while loop < 8:
            patch = image[height - PATCH_H * (loop + 1):height - PATCH_H * loop, 0:width]
            gray, blur, sobel_x, canny_x, canny = preprocess_image(patch)

            lines = houghlines(canny)
            '''
            for index, line in enumerate(lines):
                rho, theta = line

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 1)
            '''

            if lines is None or len(lines) == 0:
                break
                # print(loop)
                loop += 1
                continue

            line_left, left_lower_x, left_upper_x = nearest_lines(lines, loop == 0 and X_LEFT or left_upper_x_ll[loop - 1], PATCH_H)
            line_right, right_lower_x, right_upper_x = nearest_lines(lines, loop == 0 and X_RIGHT or right_upper_x_ll[loop - 1], PATCH_H)

            left_lower_x_ll.append(left_lower_x)
            left_upper_x_ll.append(left_upper_x)
            right_lower_x_ll.append(right_lower_x)
            right_upper_x_ll.append(right_upper_x)
            lower_y_ll.append(height - PATCH_H * (loop))
            upper_y_ll.append(height - PATCH_H * (loop + 1))

            loop += 1

        for llx, lux, rlx, rux, ly, uy in zip(left_lower_x_ll, left_upper_x_ll, right_lower_x_ll, right_upper_x_ll, lower_y_ll, upper_y_ll):
            # print(llx, lux, rlx, rux, ly, uy)
            cv2.line(mask, (int(llx), int(ly)), (int(lux), int(uy)), (0, 0, 255), 2)
            cv2.line(mask, (int(rlx), int(ly)), (int(rux), int(uy)), (0, 0, 255), 2)

        gray, _, _, _, canny = preprocess_image(image)

        cv2.imshow("Lines", mask)
        # cv2.imshow("Gray", canny)

        s = warp_bird_eye_view(image)
        cv2.imshow("warped", s)

        l = warp_bird_eye_view(mask)
        cv2.imshow("warped with lines", l)

        patch = image[height - PATCH_H * (2 + 1):height - PATCH_H * 0, 0:width]
        patch_roi = make_roi_with(preprocess_image(patch)[0], [(left_upper_x_ll[2], 0), (left_lower_x_ll[0], PATCH_H * 3), (right_lower_x_ll[0], PATCH_H * 3), (right_upper_x_ll[2], 0)])
        ppppp = make_roi_with(patch, [(left_upper_x_ll[2], 0), (left_lower_x_ll[0], PATCH_H * 3), (right_lower_x_ll[0], PATCH_H * 3), (right_upper_x_ll[2], 0)])
        if old_patch_roi is not None:
            '''
            image, flow = dense_flow(old_patch_roi, patch_roi, cv2.cvtColor(patch_roi, cv2.COLOR_GRAY2RGB))
            '''
            image, new, old = track_optical_flow(old_patch_roi, ppppp)
            cv2.imshow("Optical Flow", image)
            delta = (new - old)
            print(delta.mean(axis=0))
            '''
            flow_y = flow[:,:,1]
            flow_y[flow_y > 0] = 0
            mean = np.mean(flow[:,:,1])
            print(mean)
            '''

        old_patch_roi = patch_roi

        if cv2.waitKey(33) > 0: break
    cv2.destroyAllWindows()
