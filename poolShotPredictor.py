from ultralytics.yolo.engine.model import YOLO
import cv2
import numpy as np
import cvzone
import math
global inHole
global color

# function to select the green area that used for region of YOLOv8
def selectArea(imgArea):
    bbox = []
    rect = []
    lower = np.array([60, 70, 50])
    upper = np.array([86, 255, 255])
    hsv_img = cv2.cvtColor(imgArea, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500000:
            x, y, w, h = cv2.boundingRect(contour)
            rect.append([x, y, w, h])
            # imgArea = cv2.rectangle(imgArea, (x + 30, y + 30), (x + w - 30, y + h - 30), (0, 255, 0), 4)
            imgArea = cv2.rectangle(imgArea, (x + 30, y + 30), (x + w - 30, y + h - 30), (0, 255, 0), 4)
            holesA = [
                [x + 52, y + 52],
                [x + 52, y + h - 52],
                [x + w - 52, y + 52],
                [x + w - 52, y + h - 52],
                [x + (w - 12) // 2, y + 40],
                [x + (w - 12) // 2, y + h - 40]
            ]

            #  create the points of the holes or pockets
            for hole in holesA:
                center = hole
                radius = 50
                x = int(center[0] - radius)
                y = int(center[1] - radius)
                w = h = int(radius * 2)
                bbox.append([x, y, x + w, y + h])
                # cv2.circle(imgArea, hole, 50, (255, 0, 0), 2)

    return imgArea, bbox, rect

# function to show the results of machine learning (YOLOv8)
def machinelearning(predict, imgDetect):
    max_cue = 0
    max_white = 0
    max_color = 0
    whiteBall = []
    colorBall = []
    cuePos = []

    for r in predict:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2 - x1), (y2 - y1)

            conf = math.ceil(box.conf[0] * 100) / 100
            for c in box.cls:
                namescls = model.names[int(c)]

                if namescls == "white-ball" and conf > max_white and not whiteBall:
                    max_white = conf
                    center_x, center_y = x1 + w // 2, y1 + h // 2
                    whiteBall = [x1, y1, w, h]
                    radius = min(w, h) // 2
                    cv2.circle(imgDetect, (center_x, center_y), radius + 20, (80, 145, 75), thickness=16)
                    cvzone.putTextRect(imgDetect, f'{namescls.upper()}', (max(0, x1 + w + 20), max(50, y1 + 20)),
                                       scale=1.2, thickness=2, colorR=(0, 255, 0), offset=10)

                elif namescls == "color-ball" and conf > max_color and not colorBall:
                    max_color = conf
                    radius = min(w, h) // 2
                    colorBall = [x1, y1, w, h, radius]
                    cvzone.putTextRect(imgDetect, f'{namescls.upper()}', (max(0, x1 + w + 20), max(50, y1 + 20)),
                                       scale=1.2, thickness=2, colorR=(0, 255, 0), offset=10)

                elif namescls == "cue" and conf > max_cue and not cuePos:
                    max_cue = conf
                    center_x, center_y = x1 + w // 2, y1 + h // 2
                    if y1 > 540:
                        cuePos = [x1 + 8, y1, w, h]
                    elif y1 < 600:
                        cuePos = [x1 + 8, y1, w, h]
                    cvzone.putTextRect(imgDetect, f'{namescls.upper()}', (max(0, center_x), max(50, center_y)),
                                       scale=1.2, thickness=2, colorR=(0, 255, 0), offset=10)

    return imgDetect, whiteBall, colorBall, cuePos

# function to calculate the angle
def findAngle(deg):
    theta = math.radians(deg)
    sinus = math.sin(theta)
    cosinus = math.cos(theta)

    if abs(sinus) < 1e-15:
        cosinus = 0
    if abs(cosinus) < 1e-15:
        sinus = 0

    return sinus, cosinus

# function to show the predicted results
def showResult(paths, colorR, predictionR):
    for i, path in enumerate(paths):
        if i == 0:
            pass
        else:
            drawLine(areaSelected[0], (paths[i - 1][0], paths[i - 1][1]), (path[0], path[1]), colorR)
            cv2.circle(areaSelected[0], (path[0], path[1]), 24, colorR, cv2.FILLED)

    if predictionR:
        cvzone.putTextRect(areaSelected[0], "PREDICTION: IN", (300, 80), scale=3, thickness=4, colorR=(0, 255, 0),
                           offset=14)
    else:
        cvzone.putTextRect(areaSelected[0], "PREDICTION: OUT", (300, 80), scale=3, thickness=4, colorR=(200, 97, 64),
                           offset=14)

# function to calculate the point that cue shot the white ball
def findShotPoints(cuePos, whiteBall, radiusMeanR, shotPointsR):
    cuePoints = []
    shotPointR = []
    whiteBallX = whiteBall[0] + whiteBall[2] // 2
    whiteBallY = whiteBall[1] + whiteBall[3] // 2

    radiusMeanR.append((cuePos[2] // 2 + cuePos[3] // 2) // 2)
    radius = 0
    for i in radiusMeanR:
        radius += 1
    radius = radius // (len(radiusMeanR))

    LX = cuePos[0] + cuePos[2] // 2
    LY = cuePos[1] + cuePos[3] // 2
    for the in range(0, 360):
        sinus, cosinus = findAngle(the)
        DX = int(cosinus * radius)
        DY = int(sinus * radius)
        cuePoints.append([LX + DX, LY + DY])

    minGap = 1000000
    for cuePoint in cuePoints:
        gap = math.sqrt(math.pow(whiteBallX - cuePoint[0], 2) + math.pow(whiteBallY - cuePoint[1], 2))
        if gap < minGap:
            minGap = gap
            shotPointR = cuePoint

    shotPointsR.append(shotPointR)
    sumX = 0
    sumY = 0
    for point in shotPointsR:
        sumX += point[0]
        sumY += point[1]
    shotPointR = [sumX // len(shotPointsR), sumY // len(shotPointsR)]

    return shotPointR

# function to draw line of the ball
def drawLine(imgL, pt1, pt2, colorL):
    length = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    points = []
    for i in np.arange(0, length, 15):
        r = i / length
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        points.append(p)
    for p in points:
        for i in points:
            cv2.line(imgL, p, i, colorL, 5)

# function to calculate the line between two points
def findLine(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    try:
        m = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        m = (y2 - y1) / (x2 - x1 + 1)
    c = y1 - (m * x1)
    return m, c

# function to detect the collision between white ball and color ball
def collision(whiteBall, colorBall):
    whiteBallList = []
    colorBallList = []

    radius = (whiteBall[2] - whiteBall[0]) // 2
    LX = whiteBall[0] + (whiteBall[2] - whiteBall[0]) // 2
    LY = whiteBall[1] + (whiteBall[3] - whiteBall[1]) // 2
    for the in range(0, 360):
        sinus, cosinus = findAngle(the)
        DX = int(cosinus * radius)
        DY = int(sinus * radius)
        whiteBallList.append([LX + DX, LY + DY])

    radius = colorBall[4]
    LX = colorBall[0] + (colorBall[2] - colorBall[0]) // 2
    LY = colorBall[1] + (colorBall[3] - colorBall[1]) // 2
    for the in range(0, 360):
        sinus, cosinus = findAngle(the)
        DX = int(cosinus * radius)
        DY = int(sinus * radius)
        colorBallList.append([LX + DX, LY + DY])

    collsPoints = []
    for point in whiteBallList:
        if point in colorBallList:
            collsPoints.append(point)

    if len(collsPoints) > 0:
        xPoint = 0
        yPoint = 0
        for point in collsPoints:
            xPoint += point[0]
            yPoint += point[1]
        collsPoint = [xPoint // len(collsPoints), yPoint // len(collsPoints)]
        cv2.circle(areaSelected[0], (collsPoint[0], collsPoint[1]), 16, (80, 145, 75), cv2.FILLED)
        return True, collsPoint
    return False, []

# function to calculate the ball that will go holes or not
def bounceDetection(point, radius, holesD):
    colorD = (80, 145, 75)
    inHoleD = False
    for hole in holesD:
        p = point[0] - radius
        q = point[1] - radius
        r = point[0] + radius
        s = point[1] + radius
        if p >= hole[0] and q >= hole[1] and r <= hole[2] and s <= hole[3]:
            inHoleD = True
            colorD = (80, 145, 75)

    return colorD, inHoleD

# function to predict the direction of color ball
def pathLine(collsPoint, colorBall, paths, holesL):
    global color, inHole
    colorBallCenter = [colorBall[0] + colorBall[2] // 2, colorBall[1] + colorBall[3] // 2]
    m2, c2 = findLine(collsPoint, [colorBallCenter[0], colorBallCenter[1]])

    rectangle = selectArea(imgArea=img)
    for rects in rectangle[2]:
        print(rects)
        if collsPoint[0] > colorBall[0] + colorBall[2] // 2:
            xLast = rects[0] + 40
        else:
            xLast = rects[2] + 130

        for i in range(0, 2):
            x2 = xLast
            y2 = int((m2 * x2) + c2)

            if y2 >= rects[3] + 60:
                y2 = rects[3] + 60
                x2 = int((y2 - c2) / m2)
            if y2 <= rects[1] + 50:
                y2 = rects[1] + 50
                x2 = int((y2 - c2) / m2)
            if rects[0] + 100 < y2 < rects[3] + 10 and x2 >= rects[2] + 130:
                x2 = rects[2] + 130
                y2 = int((m2 * x2) + c2)
                xLast = rects[0] + 40
            if rects[0] + 100 < y2 < rects[3] + 10 and x2 <= rects[0] + 40:
                x2 = rects[0] + 40
                y2 = int((m2 * x2) + c2)
                xLast = rects[2] + 130

            paths.append([x2, y2])
            color, inHole = bounceDetection(paths[-1], 6, holesL)

            if inHole:
                return paths, color, inHole
            else:
                m2 = -m2
                c2 = y2 - (m2 * x2)

    return paths, color, inHole

# function to controll all calcaulations for prediction
def poolShotPrediction(shotPointS, whiteBall, colorBall, holesS):
    try:
        m1, c1 = findLine([shotPointS[0], shotPointS[1]],
                          [whiteBall[0] + whiteBall[2] // 2, whiteBall[1] + whiteBall[3] // 2])
        points = []
        xLast = (colorBall[0] + colorBall[2] // 2)
        x1, y1 = xLast, int((m1 * xLast) + c1)
        if xLast >= whiteBall[0] + whiteBall[2] // 2:
            section = 1
        else:
            section = -1

        for x in range(whiteBall[0] + whiteBall[2] // 2, xLast, section):
            y = int((m1 * x) + c1)
            points.append([x, y])

        for point in points:
            p = point[0] - whiteBall[2] // 2
            q = point[1] - whiteBall[3] // 2
            r = point[0] + whiteBall[2] // 2
            s = point[1] + whiteBall[3] // 2
            box = [p, q, r, s]
            colorBallPoint = [
                colorBall[0],
                colorBall[1],
                colorBall[0] + colorBall[2],
                colorBall[1] + colorBall[3],
                colorBall[4]
            ]
            colls, collsPoint = collision(box, colorBallPoint)

            if colls:
                x1, y1 = collsPoint[0], collsPoint[1]
                paths = [[colorBall[0] + colorBall[2] // 2, colorBall[1] + colorBall[3] // 2]]
                paths, colorS, inHoleS = pathLine(collsPoint, colorBall, paths, holesS)
                showResult(paths, colorS, inHoleS)

                xn = whiteBall[0] + whiteBall[2] // 2
                yn = whiteBall[1] + whiteBall[3] // 2
                drawLine(areaSelected[0], (xn, yn), (x1, y1), (80, 145, 75))
                cv2.circle(areaSelected[0], (x1, y1), 10, (80, 145, 75), cv2.FILLED)

                return {"prediction": inHoleS, "paths": paths, "color": colorS}

    except TypeError:
        pass

# initiliaze the video
cap = cv2.VideoCapture("Video/PoolShot.mp4")
result = cv2.VideoWriter('Video/results.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 1080))
model = YOLO("yolo-weight/pool-n.pt")

shotPoints = []
radiusMean = []
lastPoint = []
prediction = True
possibility = []
holes = []

# start the program
while True:
    success, img = cap.read()
    areaSelected = selectArea(img)

    # start the YOLOv8 to detect the objects
    results = model.predict(areaSelected[0], stream=True)
    predicted = machinelearning(results, areaSelected[0])
    holes = areaSelected[1]

    # start the calculations
    if predicted[3] and predicted[1] and predicted[2]:
        addX = predicted[1][0] + predicted[1][2] // 2
        addY = predicted[1][1] + predicted[1][3] // 2
        if not lastPoint:
            lastPoint.append([addX, addY])
            lastPoint.append([addX, addY])
        else:
            lastPoint.append([addX, addY])

        distance = lambda a, b: math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))
        if distance(lastPoint[-1], lastPoint[-2]) >= 4:
            prediction: bool = False
            probability = {}
            count = 0
            for output in possibility:
                if output is None:
                    pass
                else:
                    if possibility.count(output) > count:
                        count = possibility.count(output)
                        probability = output
            showResult(probability.get('paths', []), probability.get('color', []), probability.get('prediction', []))

        elif len(lastPoint) > 2:
            if distance(lastPoint[-2], lastPoint[-3]) >= 4 > distance(lastPoint[-1], lastPoint[-2]):
                prediction = True
                shotPoints = []
                possibility = []

        if prediction:
            shotPoint = findShotPoints(predicted[3], predicted[1], radiusMean, shotPoints)
            results = poolShotPrediction(shotPoint, predicted[1], predicted[2], holes)
            possibility.append(results)

        elif not prediction:
            showResult(probability.get('paths', []), probability.get('color', []), probability.get('prediction', []))

    frame = cv2.resize(areaSelected[0], (960, 540))
    cv2.imshow('Pool Shot Predictor', frame)
    result.write(areaSelected[0])
    key = cv2.waitKey(24)
    if key & 0xFF == ord('q'):
        break

result.release()
