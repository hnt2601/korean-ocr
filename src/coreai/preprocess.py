import cv2


def grab_contours(cnts):
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else:
        raise Exception(
            (
                "Contours tuple must have length 2 or 3, "
                "otherwise OpenCV changed their cv2.findContours return "
                "signature yet again. Refer to OpenCV's documentation "
                "in that case"
            )
        )
    return cnts


def get_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(image, 11, 17, 17)
    edged = cv2.Canny(image, 100, 255)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = image[y : y + h, x : x + w]
        break

    return roi
