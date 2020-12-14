import numpy as np
import cv2
import pickle

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

def get_bird_eye_view_transmatrix():
    src = np.float32([
        (890, 690), (1040, 690),
        (730, 1020), (1200, 1020),
    ])
    # below for RCT
    '''
        (894, 597), (1010, 598),
        (731, 989), (1197, 989),
    '''
    RATIO = 3
    width = 1920
    height = 1080 * 2
    dst = np.float32([
        (width / 2 - 30 * RATIO, height - 30 * 7 * RATIO), (width / 2 + 30 * RATIO, height - 30 * 7 * RATIO),
        (width / 2 - 30 * RATIO, height - 30 * 1 * RATIO), (width / 2 + 30 * RATIO, height - 30 * 1 * RATIO),
    ])

    return cv2.getPerspectiveTransform(src, dst)

def warp_bird_eye_view(img):
    width = 1920
    height = 1080 * 2

    transformMatrix = get_bird_eye_view_transmatrix()

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
    from datetime import datetime
    s = datetime.now()
    print(s)
    image = cv2.imread('./data/images/incheon-magnetic-1080p-left.png')

    # video_name = 'incheon-magnetic-1080p.mp4'
    video_name = 'RCT3.mp4'
    video_dir = 'data/'
    video_path = video_dir + video_name
    capture = cv2.VideoCapture(video_path) # start from 2880

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    # print(fps, width, height, total_frame)
    wrap_transform_matrix = get_bird_eye_view_transmatrix()

    speed = []
    direction = []

    # commented out
    seek = 328 # stop
    # seek = 3880 # stop
    # seek = 4000 # straight
    # seek = 13944 # right
    # seek = 24775 # left
    # seek = 28875 # left
    capture.set(cv2.CAP_PROP_POS_FRAMES, seek)

    old_patch_roi = None

    # enable when magnetic video
    '''
    sections = [
        (3890, 5483),
        (8952, 11207),
        (14720, 16908),
        (20486, 31468),
        (35254, 41477)
    ]
    current_section_index = 0
    '''

    SAVE_INTERVAL = 0
    while capture.get(cv2.CAP_PROP_POS_FRAMES) < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        '''
        if capture.get(cv2.CAP_PROP_POS_FRAMES) > sections[current_section_index][1]:
            current_section_index += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, sections[current_section_index][0])
        '''
        try:
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
            while loop < 6:
                patch = image[height - PATCH_H * (loop + 1):height - PATCH_H * loop, 0:width]
                gray, blur, sobel_x, canny_x, canny = preprocess_image(patch)

                lines = houghlines(canny)

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

            angle = []
            for llx, lux, rlx, rux, ly, uy in zip(left_lower_x_ll, left_upper_x_ll, right_lower_x_ll, right_upper_x_ll, lower_y_ll, upper_y_ll):
                # print(llx, lux, rlx, rux, ly, uy)
                cv2.line(mask, (int(llx), int(ly)), (int(lux), int(uy)), (0, 0, 255), 2)
                cv2.line(mask, (int(rlx), int(ly)), (int(rux), int(uy)), (0, 0, 255), 2)

                lt = np.array((lux, uy, 1))
                rt = np.array((rux, uy, 1))
                lb = np.array((llx, ly, 1))
                rb = np.array((rlx, ly, 1))

                wlt = np.matmul(wrap_transform_matrix, lt)
                wrt = np.matmul(wrap_transform_matrix, rt)
                wlb = np.matmul(wrap_transform_matrix, lb)
                wrb = np.matmul(wrap_transform_matrix, rb)

                wlt /= wlt[2]
                wrt /= wrt[2]
                wlb /= wlb[2]
                wrb /= wrb[2]

                pack = np.array((wlt, wrt, wlb, wrb), dtype=np.int)
                left_angle = np.arctan2(pack[0,0] - pack[2,0], pack[2,1] - pack[0,1])
                right_angle = np.arctan2(pack[1,0] - pack[3,0], pack[3,1] - pack[1,1])

                angle.append((left_angle + right_angle) / 2)

            interested_angle = angle[-4:]
            try:
                mean_angle = sum(interested_angle) / len(interested_angle)
            except ZeroDivisionError:
                mean_angle = 153123513
            # print(mean_angle / np.pi * 180)
            if old_patch_roi is not None:
                if len(direction) == 0:
                    direction.append(mean_angle)
                else:
                    last_angle = direction[-1]
                    r = last_angle / mean_angle
                    if abs(r - 1) < 0.1:
                        direction.append(last_angle)
                    else:
                        direction.append(mean_angle)

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
                image, new, old = track_optical_flow(old_patch_roi, ppppp)
                cv2.imshow("Optical Flow", image)
                delta = (new - old)
                calculated_speed = np.linalg.norm(delta.mean(axis=0))
                latest_speed = speed[-5:]
                if len(latest_speed) == 0:
                    speed.append(calculated_speed)
                else:
                    mean_speed = sum(latest_speed) / len(latest_speed)
                    r = calculated_speed / mean_speed
                    if abs(r - 1) < 0.1:
                        speed.append(calculated_speed)
                    else:
                        speed.append(mean_speed)

            old_patch_roi = patch_roi

            if cv2.waitKey(33) > 0: break
        except Exception as e:
            pass
        finally:
            if len(direction) != len(speed):
                direction = direction[:-1]
            SAVE_INTERVAL += 1

        if SAVE_INTERVAL > 100:
            with open('speed-' + video_name + '.dat', 'wb') as f:
                pickle.dump(speed, f)
            with open('direction-' + video_name + '.dat', 'wb') as f:
                pickle.dump(direction, f)
            SAVE_INTERVAL = 0

    cv2.destroyAllWindows()

    print(len(speed))
    print(len(direction))

    with open('speed-' + video_name + '.dat', 'wb') as f:
        pickle.dump(speed, f)
    with open('direction-' + video_name + '.dat', 'wb') as f:
        pickle.dump(direction, f)

    global_direction = 0.0
    D_R = 0.017
    S_R = 0.1
    current_x = 0
    current_y = 0
    positions = [(0, 0)]
    for (s, d) in zip(speed, direction):
        # print(global_direction, d, d * D_R)
        global_direction += d * D_R
        delta_x = np.cos(global_direction) * s * S_R
        delta_y = np.sin(global_direction) * s * S_R
        current_x += delta_x
        current_y += delta_y
        positions.append((current_x, current_y))

    nd_positions = np.array(positions)
    min_x, min_y = nd_positions.min(axis=0)
    max_x, max_y = nd_positions.max(axis=0)

    nd_positions -= np.array((min_x, min_y))

    MAP_WIDTH = 500
    MAP_HEIGHT = 500
    MAP_PAD = 20
    Map = np.zeros((MAP_WIDTH, MAP_HEIGHT, 3))

    prev_position = None
    MAP_SCALE = 500 / max(max_x - min_x, max_y - min_y)
    for current_position in nd_positions:
        try:
            if prev_position is not None:
                px = int((prev_position[0]) * MAP_SCALE)
                py = int((prev_position[1]) * MAP_SCALE)
                cx = int((current_position[0]) * MAP_SCALE)
                cy = int((current_position[1]) * MAP_SCALE)
                cv2.line(Map, (px, py), (cx, cy), (255, 255, 255), 1)
                cv2.line(Map, (px, py), (cx, cy), (255, 255, 255), 1)
                # cv2.circle(Map, (cx, cy), 1, (0, 0, 255), 2)

            prev_position = current_position
        except Exception as e:
            pass

    Map = np.pad(Map, ((MAP_PAD, MAP_PAD), (MAP_PAD, MAP_PAD), (0, 0)), constant_values=0)

    print(datetime.now())

    cv2.imshow("Map", Map)
    cv2.waitKey()
    cv2.destroyAllWindows()