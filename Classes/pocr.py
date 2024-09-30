from paddleocr import PaddleOCR
import re
import cv2
from Constant.constant import PATTERN

class Pocr:

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        pass

    def pocr(self, image):
        result = ""
        try:
            # Convert the NumPy array to an image format that PaddleOCR can read
            result = self.ocr.ocr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cls=True)
        except Exception as e:            
            print(f"Error during OCR processing: {e}")
            del(self.ocr)
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            result = self.ocr.ocr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cls=True)

        if result[0] is None:
            del(image)
            return "Accepted"

        rsltstr = ""
        print(f"Result: {result}\n")
        for idx in range(len(result[0])):
            res = result[0][idx]
            rsltstr += res[1][0]

        mobile_numbers_ocr = re.findall(PATTERN, rsltstr)
        if len(mobile_numbers_ocr) >= 1:
            del(image)
            return "Rejected"
        
        del(image)
        return "Accepted"