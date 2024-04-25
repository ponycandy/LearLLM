from cnocr import CnOcr
from PIL import Image
import numpy as np

img_fp = './haha.jpg'
img = Image.open('./haha.jpg')

left = 349
top = 514
right = 549
bottom = 567

# 裁剪日期
img_cropped = img.crop((left, top, right, bottom))
img_cropped.save('cropped_image_date.jpg')

left = 246
top = 644
right = 811
bottom = 762

img_cropped = img.crop((left, top, right, bottom))
img_cropped.save('cropped_image_steps.jpg')
#
# img_cropped = img_cropped.convert('L')
# ocr = CnOcr()
#
# res = ocr.ocr_for_single_line(img_cropped)
#
#
# print(res)
#
# print("hold a secs");