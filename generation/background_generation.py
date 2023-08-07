from rembg import remove
from PIL import Image
import numpy as np

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

# на инициализацию пайплайна может уйти много времени
pipe = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

TEMP_FILENAME = "temp.png"


def generate_background(filename, description, output_filename="output.jpeg"):
	"""
	Функция для генерации кринжово-рофельного заднего фона у товара.
	Принимает на вход:
		-filename: str, путь до файла с изображением товара;
		-description: str, описание товара (желательно генерировать с помощью отдельной модели или укорачивать);
		-output_filename: str, путь до выходного файла
	Удаляет задний фон с изображения и сохраняет его в output_filename.
	"""

	# удалим задний фон и сохраним как .png
	with open(filename, 'rb') as i:
		with open(TEMP_FILENAME, 'wb') as o:
			img = i.read()
			output = remove(img)
			o.write(output)

	img_array = np.array(Image.open(TEMP_FILENAME))
	mask = np.logical_not(img_array[:, :, -1]).astype(float)
	img = Image.fromarray(img_array[:, :, :3])
	prompt = "Задний фон для продажи " + description

	out = pipe(
		prompt=prompt,
		image=img,
		mask_image=mask,
		num_inference_steps=100)

	image = out.images[0]
	image.save(output_filename)