import tensorflow as tf
import urllib
import cv2
from PIL import Image
from keras.preprocessing import image
import numpy as np
import requests, time
import pathlib, os, sys
import tornado.ioloop
import tornado.web

model = tf.keras.models.load_model('my_model')
model.summary()

data_dir = pathlib.Path('clustered')
image_count = len(list(data_dir.glob('*/*.png')))

batch_size = 32
img_height = 50
img_width = 26

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


def predict_url(url):
	with urllib.request.urlopen(url) as resp:
		img = np.asarray(bytearray(resp.read()), dtype="uint8")
		return predict_img(img)
		
def predict_img(img):
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	captcha = ""

	for j in range(5):
		crop_img = img[0:50, 8+j*26:8+(j+1)*26]
		croped_img = "tmp/%s.png"%j
		cv2.imwrite(croped_img, crop_img)
		# im = Image.fromarray(np.asarray(croped_img), 'RGB')
		# img_array = np.array(im)
		# img_array = np.expand_dims(img_array, axis=0)

		img_height = 50
		img_width = 26
		char_img = image.load_img(
		    croped_img, target_size=(img_height, img_width)
		)
		
		img_array = image.img_to_array(char_img)
		img_array = tf.expand_dims(img_array, 0) # Create a batch

		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])

		captcha += class_names[np.argmax(score)]

		os.unlink(croped_img)

	return captcha

class MainHandler(tornado.web.RequestHandler):
	def get(self):
		captcha = predict_url("https://vcbdigibank.vietcombank.com.vn/w1/get-captcha/a7dac5a5-a21e-52ce-2eed-f2c44a034279")
		self.write(captcha)
	def post(self):
		file = self.request.files['file'][0]
		img = np.asarray(bytearray(file['body']), dtype="uint8")
		captcha = predict_img(img)
		self.write(captcha)

def make_app():
	return tornado.web.Application([
		(r"/", MainHandler),
	])

if __name__ == "__main__":
	app = make_app()
	app.listen(8888)
	tornado.ioloop.IOLoop.current().start()
