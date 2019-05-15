import cv2
from numpy import *
from matplotlib.pyplot import *
import pickle
import time

temp_x, temp_y = 0, 0
def detect():
    global pts1, pts2
    x1 = []
    y1 = []

    x2 = []
    y2 = []

    p21_list = []
    p22_list = []
    fin_list = []

    try:
        for i in pts1:
            #if abs(pts1[pts1.index(i)][0] - pts1[pts1.index(i)+1][0]) > 4:
            x1.append(i[0])
            #if abs(pts1[pts1.index(i)][1] - pts1[pts1.index(i)+1][1]) > 4:
            y1.append(i[1])

        for i in pts2:
            #if abs(pts2[pts2.index(i)][0] - pts2[pts2.index(i)+1][0]) > 4:
            x2.append(i[0])
            #if abs(pts2[pts2.index(i)][1] - pts2[pts2.index(i)+1][1]) > 4:
            y2.append(i[1])

        if len(x1) > 0 and len(y1) > 0:
            p21 = polyfit(x1, y1, 4)
            for i in p21:
                p21_list.append(i)

        if len(x2) > 0 and len(y2) > 0:
            p22 = polyfit(x2, y2, 4)
            for i in p22:
                p22_list.append(i)

        fin_list = p21_list + p22_list

        if len(fin_list) < 10:
            for i in range(5):
                fin_list.append(0)
    except:
        pass

    try:
        filename = "Gestures.sav"
        pickle_in = open(filename, "rb")
        loaded_model = pickle.load(pickle_in)

        prediction = loaded_model.predict([fin_list])  # Coefficients of polynomial

        if prediction == 0:
            print("Hello")

        elif prediction == 1:
            print("Bag")

        elif prediction == 2:
            print("Chair")

        elif prediction == 3:
            print("Thank You")

        elif prediction == 4:
            print("What time is it?")

        elif prediction == 5:
            print("Please")

        elif prediction == 6:
            print("May I go to washroom?")

        elif prediction == 7:
            print("Exam")

        elif prediction == 8:
            print("Pencil")

        else:
            print("Didn't recognize!")

        pts1 = []
        pts2 = []

    except:
        pass

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(1)

pts1 = []
pts2 = []

if cam.isOpened():
    ret, frame = cam.read()
else:
    ret = False

_, frame1 = cam.read()
_, frame2 = cam.read()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
x_face, y_face = 0, 0


while True:
    ycrcb_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
    ycrcb_frame = cv2.GaussianBlur(ycrcb_frame, (19, 19), 2)
    # ycrcb_frame = cv2.medianBlur(frame1, 11)

    # Splitting & smoothing the channels
    y, cr, cb = cv2.split(ycrcb_frame)

    # Processing the channels
    mask_y = cv2.inRange(y, 27, 113)
    mask_cr = cv2.inRange(cr, 132, 172)
    mask_cb = cv2.inRange(cb, 113, 142)
    # mask_y = cv2.inRange(y, 61, 113)
    # mask_cr = cv2.inRange(cr, 138, 151)
    # mask_cb = cv2.inRange(cb, 109, 128)

    # Merging the channels
    res = cv2.bitwise_and(mask_y, mask_cr)
    res = cv2.bitwise_and(res, mask_cb)
    kernel = np.ones((7, 7), np.uint8)

    # Morphological operations
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    # Background removal
    bg = cv2.bitwise_and(frame1, frame1, mask=opening)

    # Face detection
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 2)

    mask = np.ones((480, 640), dtype=np.uint8)
    mask = cv2.bitwise_not(mask)
    kernel_face = np.ones((19, 19), np.uint8)

    for (x, y, w, h) in faces:
        if len(faces) < 2:
            x -= 30
            y -= 50
            w += 150
            h += 60
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 3)
            # cv2.dilate(mask, kernel_face)
            roi_corners = np.array([[(x, y), (x + h, y), (x + h, y + w), (x, y + w)]], dtype=np.int32)
            cv2.fillPoly(mask, roi_corners, (0, 0, 0))

    bg_bin = cv2.bitwise_and(opening, opening, mask=mask)
    bg_bin = cv2.morphologyEx(bg_bin, cv2.MORPH_CLOSE, kernel)

    # Finding contours
    contours, _ = cv2.findContours(bg_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    extracted_contours = []
    all_y = []
    to_remove = 0

    for contour in contours:
        a, b, w1, h1 = cv2.boundingRect(contour)
        rect_area = w1 * h1
        ac = a + w1/2
        bc = b + h1/2
        center_rect = (int(ac), int(bc))

        if rect_area > 9000 * 0.25:
            extracted_contours.append((a, b, w1, h1))
            all_y.append(b)

            try:
                if len(faces) > 0:
                    cv2.putText(bg, "FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    for (x, y, w, h) in faces:
                        # cv2.rectangle(bg, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        if a < x:
                            if abs(temp_x - int(ac)) > 5 or abs(temp_y - int(bc)) > 5:
                                temp_x = int(ac)
                                temp_y = int(bc)
                                pts1.append(center_rect)
                        elif a > x:
                            if abs(temp_x - int(ac)) > 5 or abs(temp_y - int(bc)) > 5:
                                temp_x = int(ac)
                                temp_y = int(bc)
                                pts2.append(center_rect)

                        #print(pts1, pts2)

                    for i in range(1, len(pts1) - 1):
                        # if either of the tracked points are None, ignore them
                        if pts1[i - 1] is None or pts1[i] is None:
                            continue
                        pts1_rev = pts1[-1:0:-1]
                        # otherwise, compute the thickness of the line and draw the connecting lines
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(bg, pts1_rev[i - 1], pts1_rev[i], (0, 255, 0), thickness)
                        # print(len(pts1))

                        if thickness < 3:
                            for j in pts1:
                                pts1.remove(j)

                    for i in range(1, len(pts2) - 1):
                        # if either of the tracked points are None, ignore them
                        if pts2[i - 1] is None or pts2[i] is None:
                            continue
                        pts2_rev = pts2[-1:0:-1]
                        # otherwise, compute the thickness of the line and draw the connecting lines
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(bg, pts2_rev[i - 1], pts2_rev[i], (0, 255, 0), thickness)

                        if thickness < 3:
                            for j in pts2:
                                pts2.remove(j)

                else:
                    cv2.putText(bg, "WAIT FOR FACE RECOGNITION", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                pass

    try:
        to_remove = all_y.index(min(all_y))
    except ValueError:
        pass
    for i in range(len(extracted_contours)):
        if len(all_y) > 2:
            if to_remove != i:
                a = extracted_contours[i][0]
                b = extracted_contours[i][1]
                w1 = extracted_contours[i][2]
                h1 = extracted_contours[i][3]
                cv2.rectangle(bg, (a, b), (a + w1, b + h1), (255, 0, 0), 3)
        else:
            a = extracted_contours[i][0]
            b = extracted_contours[i][1]
            w1 = extracted_contours[i][2]
            h1 = extracted_contours[i][3]
            cv2.rectangle(bg, (a, b), (a + w1, b + h1), (255, 0, 0), 3)


    frame1 = frame2
    _, frame2 = cam.read()
    frame2 = cv2.flip(frame2, 1)

    cv2.imshow("Mask", mask)
    cv2.imshow("Bg", bg)

    if len(pts1) > 19 or len(pts2) > 19:
        detect()
        time.sleep(1)
        pts1 = []
        pts2 = []

    if cv2.waitKey(1) & 0xFF == ord("a"):
        pts1 = []
        pts2 = []

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
