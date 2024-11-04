import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd

record_data = 0
data = []
i = 0
record_conf = 1
blink = 0
minAngle = 180
maxAngle = 0
record_angle = 90
prev_cx = 0
buffer_width = 50
reverse = False
prevAngle = 90
temp_record = []
condition = 0
prev_condition = 0
prev_record_angle = 0
angle = 0
cx = 0


def do_nothing(x):
    pass


def rotate_image(image, sudut):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, sudut, 1.0)
    hasil = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return hasil

# Usage: python3 main.py video_filename fps_playback
if __name__ == '__main__':
    # reading the argument source vid
    video = sys.argv[1]
    speed = sys.argv[2]

    try:
        video = int(video)
    except ValueError:
        video = str(video)
    if isinstance(video, int):
        waitTime = 1
    else:
        waitTime = int(1000 / int(speed))

    # reading the video
    source = cv2.VideoCapture(video)
    frame_width = int(source.get(3))
    frame_height = int(source.get(4))

    # loading the conf file
    try:
        with open('objs.pkl', 'rb') as f:
            hue_min, hue_max, sat_min, sat_max, val_min, val_max, rot, contrast, brightness, blur, xMin, xMax, yMin, \
            yMax, textX, textY, x, trigger, debug, record_conf, buffer_width, circlePosX, circlePosY, circleSize, \
            circle2PosX, circle2PosY, circle2Size = pickle.load(f)
    except:
        print("Configuration not loaded")
        hue_min, hue_max, sat_min, sat_max, val_min, val_max, rot, contrast, brightness, blur, xMin, xMax, yMin, \
        yMax, textX, textY, x, trigger, debug, record_conf, buffer_width, circlePosX, circlePosY, circleSize, \
        circle2PosX, circle2PosY, circle2Size = 0, 255, 0, 255, 0, 103, -3, 3, 20, 2, 0, frame_width, 0, \
                                                frame_height, 448, 416, 132, 0, 0, 1, 50, 0, 0, 100, 0, 0, 100

    if debug:
        cv2.namedWindow("Slider", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Hue Min", "Slider", hue_min, 255, do_nothing)
        cv2.createTrackbar("Hue Max", "Slider", hue_max, 255, do_nothing)
        cv2.createTrackbar("Saturation Min", "Slider", sat_min, 255, do_nothing)
        cv2.createTrackbar("Saturation Max", "Slider", sat_max, 255, do_nothing)
        cv2.createTrackbar("Value Min", "Slider", val_min, 255, do_nothing)
        cv2.createTrackbar("Value Max", "Slider", val_max, 255, do_nothing)
        cv2.createTrackbar("Rotation", "Slider", rot, 90, do_nothing)
        cv2.setTrackbarMin('Rotation', 'Slider', -90)
        cv2.createTrackbar("Contrast", "Slider", contrast, 10, do_nothing)
        cv2.createTrackbar("Brightness", "Slider", brightness, 100, do_nothing)
        cv2.createTrackbar("Gaussian Blur", "Slider", blur, 20, do_nothing)
        cv2.createTrackbar("Rect. Area", "Slider", trigger, 5000, do_nothing)
        cv2.createTrackbar("XMin", "Slider", xMin, frame_width, do_nothing)
        cv2.createTrackbar("XMax", "Slider", xMax, frame_width, do_nothing)
        cv2.createTrackbar("YMin", "Slider", yMin, frame_height, do_nothing)
        cv2.createTrackbar("YMax", "Slider", yMax, frame_height, do_nothing)
        cv2.createTrackbar("TextX", "Slider", textX, frame_width, do_nothing)
        cv2.createTrackbar("TextY", "Slider", textY, frame_height, do_nothing)
        cv2.createTrackbar("CenterX", "Slider", x, frame_width, do_nothing)
        cv2.createTrackbar("Separator", "Slider", record_conf, frame_width, do_nothing)
        cv2.createTrackbar("BufWidth", "Slider", buffer_width, 500, do_nothing)

        cv2.namedWindow("Circle Configuration", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("CircleX", "Circle Configuration", circlePosX, frame_width, do_nothing)
        cv2.createTrackbar("CircleY", "Circle Configuration", circlePosY, frame_height, do_nothing)
        cv2.createTrackbar("CircleSize", "Circle Configuration", circleSize, frame_width, do_nothing)
        cv2.createTrackbar("Circle2X", "Circle Configuration", circle2PosX, frame_width, do_nothing)
        cv2.createTrackbar("Circle2Y", "Circle Configuration", circle2PosY, frame_height, do_nothing)
        cv2.createTrackbar("Circle2Size", "Circle Configuration", circle2Size, frame_width, do_nothing)

    size = (frame_width, frame_height)
    print(size)

    # running the loop
    while True:
        # extract the values from the trackbar
        if debug:
            hue_min = cv2.getTrackbarPos("Hue Min", "Slider")
            hue_max = cv2.getTrackbarPos("Hue Max", "Slider")
            sat_min = cv2.getTrackbarPos("Saturation Min", "Slider")
            sat_max = cv2.getTrackbarPos("Saturation Max", "Slider")
            val_min = cv2.getTrackbarPos("Value Min", "Slider")
            val_max = cv2.getTrackbarPos("Value Max", "Slider")
            rot = cv2.getTrackbarPos("Rotation", "Slider")
            contrast = cv2.getTrackbarPos("Contrast", "Slider")
            brightness = cv2.getTrackbarPos("Brightness", "Slider")
            blur = cv2.getTrackbarPos("Gaussian Blur", "Slider")
            trigger = cv2.getTrackbarPos("Rect. Area", "Slider")
            xMin = cv2.getTrackbarPos("XMin", "Slider")
            xMax = cv2.getTrackbarPos("XMax", "Slider")
            yMin = cv2.getTrackbarPos("YMin", "Slider")
            yMax = cv2.getTrackbarPos("YMax", "Slider")
            textX = cv2.getTrackbarPos("TextX", "Slider")
            textY = cv2.getTrackbarPos("TextY", "Slider")
            x = cv2.getTrackbarPos("CenterX", "Slider")
            record_conf = cv2.getTrackbarPos("Separator", "Slider")
            buffer_width = cv2.getTrackbarPos("BufWidth", "Slider")
            circleSize = cv2.getTrackbarPos("CircleSize", "Circle Configuration")
            circlePosY = cv2.getTrackbarPos("CircleY", "Circle Configuration")
            circlePosX = cv2.getTrackbarPos("CircleX", "Circle Configuration")
            circle2Size = cv2.getTrackbarPos("Circle2Size", "Circle Configuration")
            circle2PosY = cv2.getTrackbarPos("Circle2Y", "Circle Configuration")
            circle2PosX = cv2.getTrackbarPos("Circle2X", "Circle Configuration")

        # extracting the frames
        ret, frame0 = source.read()
        if ret:

            frame0 = rotate_image(frame0, rot)
            frame0 = cv2.circle(frame0, (circlePosX, circlePosY), circleSize, (255, 255, 255), -1)
            frame0 = cv2.circle(frame0, (circle2PosX, circle2PosY), circle2Size, (255, 255, 255), -1)

            try:
                frame = frame0[yMin:yMax, xMin:xMax]
                frame = cv2.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, brightness)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            except:
                frame = frame0
                frame = cv2.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, brightness)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Threshold of black in HSV space
            lower_black = np.array([hue_min, sat_min, val_min])
            upper_black = np.array([hue_max, sat_max, val_max])

            # preparing the mask to overlay
            mask = cv2.inRange(hsv, lower_black, upper_black)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = np.uint8(result)

            # Gaussian and dilate filtering
            result = cv2.GaussianBlur(result, (2 * blur + 1, 2 * blur + 1), cv2.BORDER_DEFAULT)
            result = cv2.dilate(result, (1, 1), iterations=2)

            # otsu thresholding
            ret3, thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Threshold", thresh)

            # Finding Conture of detected Rectangle
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            try:
                if trigger != 0:
                    for j in range(len(contours)):
                        if cv2.contourArea(contours[j]) < trigger:
                            contours.pop(j)

                cnt = contours[0]
                rotatedRect = cv2.minAreaRect(cnt)
                (cx, cy), (width, height), angle = rotatedRect

                # centroid of the rectangle conture
                cx = int(cx)
                cy = int(cy)

                if width > height:
                    angle = angle + 90
                else:
                    angle = angle

                if cx < x and angle <= 90:
                    angle = angle + 180

                deltaAngle = abs(angle - prevAngle)

                if cx > record_conf and deltaAngle < 0.5:
                    if not reverse:
                        record_angle = min([angle, prevAngle])
                    else:
                        record_angle = max([angle, prevAngle])
                elif cx < record_conf and deltaAngle < 0.5:
                    if not reverse:
                        record_angle = max([angle, prevAngle])
                    else:
                        record_angle = min([angle, prevAngle])

                prevAngle = angle

                # Draw rectangle around the mask
                im = cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)
                cv2.circle(im, (cx, cy), 2, (200, 255, 0), 2)

                # create recording circle blink
                if record_data and not i % 15:
                    if blink:
                        blink = 0
                    else:
                        blink = 1
                if record_data and blink:
                    cv2.circle(im, (10, 10), 4, (0, 0, 255), 5)  # draw center

                # min and max Angle
                if angle < minAngle:
                    minAngle = angle
                if angle > maxAngle:
                    maxAngle = angle

                # description
                cv2.putText(im, str("Angle: " + str(round(angle))), (textX, textY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                (_, label_height), _ = cv2.getTextSize(str("Angle: " + str(int(angle))),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(im, str("Record data (r for reverse): ") + str(round(record_angle)),
                            (textX, textY + label_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if record_data:
                    cv2.putText(im, str("Recording data..."), (textX, textY + 2 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(im, str("Press 'space' to stop"), (textX, textY + 3 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(im, str("MinAngle: ") + str(int(minAngle)), (textX, textY + 4 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(im, str("MaxAngle: ") + str(int(maxAngle)), (textX, textY + 5 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(im, str("Press 'space' to record"), (textX, textY + 2 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(im, str("MinAngle: ") + str(int(minAngle)), (textX, textY + 3 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(im, str("MaxAngle: ") + str(int(maxAngle)), (textX, textY + 4 * label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if record_data:
                    if cx > record_conf and abs(prev_record_angle - record_angle) < 1:
                        temp_record.append(record_angle)
                        condition = 0
                    elif cx < record_conf and abs(prev_record_angle - record_angle) < 1:
                        temp_record.append(record_angle)
                        condition = 1
                    if condition == 1 and prev_condition == 0:
                        # print(str(min(temp_record)) + " , " + str(temp_record))
                        data.append(min(temp_record))
                        temp_record = []
                    elif condition == 0 and prev_condition == 1:
                        # print(str(max(temp_record)) + " , " + str(temp_record))
                        data.append(max(temp_record))
                        temp_record = []
                    prev_condition = condition
                    if deltaAngle < 0.5:
                        temp_record.append(record_angle)
                    elif deltaAngle > 0.5 and abs(prev_cx - cx) > 4:
                        temp_record = []
            except:
                im = frame
                # creating recording circle blink
                if record_data and not i % 15:
                    if blink:
                        blink = 0
                    else:
                        blink = 1
                if record_data and blink:
                    cv2.circle(im, (10, 10), 4, (0, 0, 255), 5)  # draw center

                    # description
                    cv2.putText(im, str("Angle: " + str(round(angle))), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                    (_, label_height), _ = cv2.getTextSize(str("Angle: " + str(int(angle))),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.putText(im, str("Record data (r for reverse): ") + str(round(record_angle)),
                                (textX, textY + label_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    if record_data:
                        cv2.putText(im, str("Recording data..."), (textX, textY + 2 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(im, str("Press 'space' to stop"), (textX, textY + 3 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(im, str("MinAngle: ") + str(int(minAngle)), (textX, textY + 4 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(im, str("MaxAngle: ") + str(int(maxAngle)), (textX, textY + 5 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(im, str("Press 'space' to record"), (textX, textY + 2 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(im, str("MinAngle: ") + str(int(minAngle)), (textX, textY + 3 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(im, str("MaxAngle: ") + str(int(maxAngle)), (textX, textY + 4 * label_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if debug:
                # trigger line
                cv2.line(im, (x, 0), (x, frame_height), (0, 255, 0), thickness=2)
                cv2.line(im, (record_conf, 0), (record_conf, frame_height), (255, 0, 0), thickness=2)
                cv2.line(im, (record_conf + buffer_width, 0), (record_conf + buffer_width, frame_height), (255, 0, 0),
                         thickness=2)
                cv2.line(im, (record_conf - buffer_width, 0), (record_conf - buffer_width, frame_height), (255, 0, 0),
                         thickness=2)

            cv2.namedWindow('Detected Rect', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Detected Rect', im)
            key = cv2.waitKey(waitTime)
            if key == 27:  # esc key
                break
            elif key == 114:  # r key
                reverse = not reverse
            elif key == 32:  # space key
                if record_data == 0:
                    record_data = 1
                else:
                    record_data = 0
                    # dumping recorded data
                    if len(data) > 0:
                        try:
                            df = pd.DataFrame(data, columns=['Angle'])
                            df.to_excel("output.xlsx")
                        except:
                            df = pd.DataFrame(data, columns=['Angle'])
                            df.to_excel("newoutput.xlsx")
                        data = []
            elif key == 100 and not record_data:  # restart with debug/normal
                if debug:
                    debug = 0
                else:
                    debug = 1
                # Saving the objects:
                with open('objs.pkl', 'wb') as f:
                    pickle.dump(
                        [hue_min, hue_max, sat_min, sat_max, val_min, val_max, rot, contrast, brightness, blur, xMin,
                         xMax, yMin, yMax, textX, textY, x, trigger, debug, record_conf, buffer_width, circlePosX,
                         circlePosY, circleSize, circle2PosX, circle2PosY, circle2Size], f)
                # dumping recorded data
                if len(data) > 0:
                    # print(data)
                    try:
                        df = pd.DataFrame(data, columns=['Angle'])
                        df.to_excel("output.xlsx")
                    except:
                        df = pd.DataFrame(data, columns=['Angle'])
                        df.to_excel("newoutput.xlsx")
                    data = []
                os.execv(sys.executable, [sys.argv[0], str(video), speed])
            prev_cx = cx
            prev_record_angle = record_angle
        else:
            break
    cv2.destroyAllWindows()
    source.release()

    # Saving the objects:
    with open('objs.pkl', 'wb') as f:
        pickle.dump([hue_min, hue_max, sat_min, sat_max, val_min, val_max, rot, contrast, brightness, blur, xMin,
                     xMax, yMin, yMax, textX, textY, x, trigger, debug, record_conf, buffer_width, circlePosX,
                     circlePosY, circleSize, circle2PosX, circle2PosY, circle2Size], f)

    # dumping recorded data
    if len(data) > 0:
        # print(data)
        try:
            df = pd.DataFrame(data, columns=['Angle'])
            df.to_excel("output.xlsx")
        except:
            df = pd.DataFrame(data, columns=['Angle'])
            df.to_excel("newoutput.xlsx")
        data = []
