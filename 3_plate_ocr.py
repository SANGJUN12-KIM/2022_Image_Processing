from plateocr import image2text
from plateocr import ocrModelConfig

reader = ocrModelConfig.model(custom_model=True)

text = image2text.read_text_area(reader, input_file='car_num_img/semple_3.jpeg')

print(text)
