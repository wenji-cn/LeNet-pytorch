from PIL import Image,ImageDraw,ImageFont

font=ImageFont.truetype("arial.ttf",22)
for digit in range(10):
	img=Image.new("L",(28,28),0)
	draw=ImageDraw.Draw(img)
	draw.text((8,2),str(digit),fill=255,font=font)
	img.save(f"./Img_data/test_{digit}.png")

print("完成图像生成")