import numpy as np
import cv2
import math

def main():
    print('Hello, Wolrd!')

    video_name = 'incheon-magnetic.mp4'
    # video_name = 'RCT3-360p.mp4'
    video_dir = 'data/'
    video_path = video_dir + video_name
    capture = cv2.VideoCapture(video_path) # start from 2880

    capture.open(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    # for _ in range(14744): # skip
    for _ in range(2880): # skip
    # for _ in range(321): # skip
        capture.read()

    print(total_frame)

    while capture.get(cv2.CAP_PROP_POS_FRAMES) < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.read()
        capture.read()
        ret, frame = capture.read()

        # set ROI
        mask = np.zeros(frame.shape, np.uint8)
        points = np.array([
            [width // 2, height * 1 // 4],
            [(width) * (0.5 + 0.3), height],
            [(width) * (0.5 - 0.3), height],
        ], np.int32)
        points = points.reshape((-1, 1, 2))
        mask_line = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask = cv2.fillPoly(mask_line.copy(), [points], (255, 255, 255))

        ROI = cv2.bitwise_and(mask, frame)

        cv2.imshow("ROI", ROI)

        # get gray color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        RRR = cv2.bitwise_and(mask_gray, edges)

        img2 = np.zeros(frame.shape)

        # http://www.gisdeveloper.co.kr/?p=6714
        lines = cv2.HoughLines(RRR, 1, np.pi / 180, 100)
        for index, line in enumerate(lines):
            rho, theta = line[0]
            degree = theta / math.pi * 180
            if degree > 30 and degree < 150:
                continue

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if index < 5:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.line(img2, (x1, y1), (x2, y2), (255, 255, 255), 1)

        '''
        minLineLength = 50
        maxLineGap = 10

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        '''

        # get lines

        # get vanishing point
        # https://stackoverflow.com/questions/57535865/extract-vanishing-point-from-lines-with-open-cv
        # delete lines
        kernel = np.ones((3,3),np.uint8)
        img2 = cv2.erode(img2,kernel,iterations = 1)
        # strengthen intersections
        kernel = np.ones((5,5),np.uint8)
        img2 = cv2.dilate(img2,kernel,iterations = 1)
        # close remaining blobs
        kernel = np.ones((9,9),np.uint8)
        img2 = cv2.erode(img2,kernel,iterations = 1)
        img2 = cv2.dilate(img2,kernel,iterations = 1)
        cv2.imshow('points.jpg', img2)

        # get rails

        # projection

        # calculate

        # get speed from optical flow

        cv2.imshow("VideoFrame", frame)
        # cv2.imshow("Edges", edges)
        cv2.imshow("Edges", RRR)

        # 전진

        if cv2.waitKey(33) > 0: break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
