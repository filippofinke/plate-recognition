import cv2
from os import listdir
from os.path import isfile, join
import random
import pytesseract
import numpy as np
IMAGES_PATH = './images/'

images = [f for f in listdir(IMAGES_PATH) if isfile(join(IMAGES_PATH, f))]
print("Found " + str(len(images)) + " images!")

def filter(input):
    valids = []
    for character in input:
        if character.isalpha() or character.isdigit():
            valids.append(character)
    return ''.join(valids)

for image in images:

    original = cv2.imread(IMAGES_PATH + image)

    grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(grayscale, 140,255)
    grayscale[mask != 0] = 255
    grayscale[mask == 0] = 0

    contours,h = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for index, contour in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 4:
            if h < w and w/h > 2 and w > 100:
                plate = grayscale[y:y+h, x:x+w]
                text = pytesseract.image_to_string(plate)
                if text != "":
                    plate_text = filter(text)
                    cv2.imshow(plate_text, plate)
                    cv2.rectangle(original, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(original, plate_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    cv2.putText(original, "@filippofinke",(20, original.shape[0] - 20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.imshow('Image' ,original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()