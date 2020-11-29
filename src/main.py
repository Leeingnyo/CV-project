import numpy as np
import cv2

def main():
    print('Hello, Wolrd!')

    video_name = 'incheon-magnetic.mp4'
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
        capture.read()

    print(total_frame)

    while capture.get(cv2.CAP_PROP_POS_FRAMES) < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray", gray)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        cv2.imshow("VideoFrame", frame)
        cv2.imshow("Edges", edges)

        if cv2.waitKey(33) > 0: break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
