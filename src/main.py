import numpy as np
import cv2
import math
from PIL import Image, ImageDraw
from detect_rail_lane import preprocess_image, dense_flow, make_roi

def main():
    print('Hello, World!')

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

    for _ in range(14744): # skip
    # for _ in range(2880): # skip
    # for _ in range(321): # skip
        capture.read()

    print(total_frame)

    ret, old_frame = capture.read()
    old_gray = preprocess_image(old_frame)[0]

    maparr = np.zeros((1,1))
    idx = [0,0]
    prevmapindex = [1000,20000]
    mapindex = [1000,20000]
    # img = Image.new("RGB", (40000, 40000))
    # img1 = ImageDraw.Draw(img)
    count = 0
    i=0
    while capture.get(cv2.CAP_PROP_POS_FRAMES) < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.read()
        capture.read()
        ret, frame = capture.read()
        gray, _, _, canny_x, _ = preprocess_image(frame)

        mask, flow = dense_flow(old_gray, gray, frame)
        speed = np.sum(np.sum(flow,axis=0),axis=0)/flow.size
        idx[0] += speed[1]
        idx[1] += speed[0]
        idx = np.around(idx)
        # img1.line([tuple(prevmapindex), tuple(mapindex)], fill ="red", width = 0) 
        # prevmapindex = mapindex

        # if count == 1000:
        #     img.save(f"asdf_{i}.jpg")
            # img = Image.new("RGB", (1000, 1000))
            # img1 = ImageDraw.Draw(img)
            # prevmapindex = [500,500]
            # mapindex=[500,500]
            # i+=1
            # count = 0

        x = int(idx[0])
        y = int(idx[1])
        try:
            if x < 0:
                maparr = np.append(np.zeros( (1, maparr.shape[1]) ), maparr, axis=0)
                x += 1
            elif x >= maparr.shape[0]:
                maparr = np.append(maparr, np.zeros( (x - maparr.shape[0] + 1, maparr.shape[1]) ), axis=0)
            if y < 0:
                maparr = np.append(np.zeros( (maparr.shape[0], 1) ), maparr, axis=1)
                y += 1
            elif y >= maparr.shape[1]:
                maparr = np.append(maparr, np.zeros((maparr.shape[0], y - maparr.shape[1] + 1)), axis=1)
            maparr[x][y] = 255
        except:
            print(maparr.shape, x, y)
        # np.savetxt('asdf.txt',maparr,fmt="%d")
        count += 1
        print(count, maparr.shape)
        if count == 9315:
            img=Image.fromarray(maparr)
            img=img.convert('1')
            np.savetxt('result.txt',maparr,fmt="%d")
            img.save("result.png")

        # http://www.gisdeveloper.co.kr/?p=6714
        lines = cv2.HoughLines(make_roi(canny_x), 1, np.pi / 180, 100)
        if lines is not None:
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

                '''
                if index < 5:
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    # cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 1)
                '''
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("Canny X", canny_x)
        cv2.imshow("Flow", mask)

        old_gray = gray
        old_frame = frame

        '''
        minLineLength = 50
        maxLineGap = 10

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        '''

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
        '''

        if cv2.waitKey(33) > 0: break
    img=Image.fromarray(maparr)
    img=img.convert('1')
    np.savetxt('result.txt',maparr,fmt="%d")
    img.save("result.png")
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
