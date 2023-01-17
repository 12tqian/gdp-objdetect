from PIL import Image, ImageDraw

image = Image.new('RGB', (200, 200))
draw = ImageDraw.Draw(image)
draw.ellipse((0, 0, 200, 200), fill = 'blue', outline ='blue')
draw.point((100, 100), 'red')
image = image.convert("RGB")
#image.save('/mnt/tcqian/danielxu/synthetic_data_test/test.jpg')
image.save('test.jpg')