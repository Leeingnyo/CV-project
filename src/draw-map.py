import numpy as np
import cv2
import pickle

if __name__ == "__main__":
    video_name = 'incheon-magnetic-1080p.mp4'
    # video_name = 'RCT3.mp4'

    with open('speed-' + video_name + '.dat', 'rb') as f:
        speed = pickle.load(f)
    with open('direction-' + video_name + '.dat', 'rb') as f:
        direction = pickle.load(f)

    global_direction = 0.0
    D_R = 0.003
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

    cv2.imshow("Map", Map)
    cv2.waitKey()
    cv2.destroyAllWindows()