import argparse
from PIL import Image
import numpy as np
import math
import pytesseract
import cv2
import os


# Pre-processing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='Data/test/0004.jpg',
                        help='Image Path')

    args = parser.parse_args()

    print("Image:", args.image_path)

    img = cv2.imread(args.image_path)

    # noise reduction
    img_c = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # grayscale
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    # Sobel
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

    # binary
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # Hough Lines
    hough = binary.astype(np.uint8)
    lines = cv2.HoughLinesP(hough, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=100)

    # for line in lines:
    #     cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2)

    k_dict = {}
    k = 0

    # 求出所有直线斜率，求出众数（考虑误差）
    for line in lines:
        if line[0][2] - line[0][0] == 0:
            continue
        # print(line[0][3], line[0][1], line[0][2], line[0][0])
        k = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
        # α = atan(k) * 180 / PI
        k = math.atan(k) * 180 / np.pi
        if len(k_dict.keys()) == 0:
            k_dict[k] = 1
        else:
            flag = False
            for item in k_dict.keys():
                if abs(item - k) < 2:
                    flag = True
                    k_dict[item] += 1
                    break
            if not flag:
                k_dict[k] = 1

    must_k_num = 0
    must_key = 0
    for item in k_dict.keys():
        if k_dict[item] > must_k_num:
            must_k_num = k_dict[item]
            must_key = item

    # print(must_key)

    # rotate the image

    h, w = img.shape[:2]
    add_w = int((((w * w + h * h) ** 0.5) - w) / 2)
    add_h = int((((w * w + h * h) ** 0.5) - h) / 2)
    # print(add_w, add_h)

    img = cv2.copyMakeBorder(img, add_h, add_h, add_w, add_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, must_key, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    cv2.imwrite(args.image_path, rotated)

    # cv2.imshow("rotated", rotated)
    # cv2.waitKey(0)

    # OCR
    preprocess = "thresh"
    # preprocess = "blur"

    image = cv2.imread(args.image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply thresholding to segment the foreground from the background
    # 这种阈值方法对于读取覆盖在灰色形状上的暗文本非常有用
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # median blurring to remove salt and pepper noise
    elif preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    # save the grayscale image as a temporary file so we can apply OCR to it
    filename = os.path.join("data/test", "postprocess.png")
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, then delete the temporary file
    # convert the contents of the image into string
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print(text)

    # # show the output images
    # cv2.imshow("Image", image)
    # cv2.imshow("Output", gray)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
