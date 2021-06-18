import cv2
for i in range(1000):
	img = cv2.imread("original/%s.png"%i)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for j in range(5):
		crop_img = gray[0:50, 8+j*26:8+(j+1)*26]
		cv2.imwrite("splited/%s_%s.png"%(i, j), crop_img)
	print(i)
