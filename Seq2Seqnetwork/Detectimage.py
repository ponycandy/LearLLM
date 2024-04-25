from cnocr import CnOcr

from datetime import datetime

today = datetime.now()
month = today.month
day = today.day
formatted_date = today.strftime("%m月%d日").lstrip("0").replace("月0", "月")

print("今天的日期是：{}".format(formatted_date))




img_fp = './cropped_image_date.jpg'
ocr = CnOcr()  # 所有参数都使用默认值
out = ocr.ocr(img_fp)

print(out[0].get('text'))

if formatted_date in out[0].get('text'):
    print("日期校验正确")
else:
    print("日期校验失败")

img_fp = './cropped_image_steps.jpg'

out = ocr.ocr(img_fp)

print(out[0].get('text'))
num = int(out[0].get('text'))
print(num)

if num > 3000:
    print("完成今日步数")
else:
    print("未完成今日步数")