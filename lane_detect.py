import cv2
import numpy as np

def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(gray, 50, 150)
    return canny


def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[
        (200, height),
        (width, 350),
        (1200, height), ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100,
                           np.array([]), minLineLength=40, maxLineGap=5)


def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


# def display_lines(img, lines):
#     line_image = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
#     return line_image

# def display_lines(img, lines):
#     line_image = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             if line is not None:
#                 for x1, y1, x2, y2 in line:
#                     print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
#                     cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)
#     return line_image

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None and len(lines) == 2:
        line1, line2 = lines[0], lines[1]
        if line1 is not None and line2 is not None and len(line1) == 1 and len(line2) == 1:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

            # Create the box using the intersection points
            box_top_left = (min(x1, x2, x3, x4), min(y1, y2, y3, y4))
            box_bottom_right = (max(x1, x2, x3, x4), max(y1, y2, y3, y4))

            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
            cv2.line(line_image, (x3, y3), (x4, y4), (0, 0, 255), 10)
            cv2.rectangle(line_image, box_top_left, box_bottom_right, (0, 255, 0), 10)

    return line_image

def make_points(image, line):
    try:
        slope, intercept = line
    except TypeError:
        return None
    # slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    # cv2.imshow("cropped_canny",cropped_canny)

    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    cv2.imshow("result", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
