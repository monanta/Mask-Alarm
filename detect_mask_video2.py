# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# ambil dimensi bingkai dan kemudian buat gumpalan
	# dari itu
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# lewati gumpalan melalui jaringan dan dapatkan deteksi wajah
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# inisialisasi daftar wajah kami, lokasinya yang sesuai,
	# dan daftar prediksi dari jaringan masker wajah kami
	faces = []
	locs = []
	preds = []

	# ulangi deteksi
	for i in range(0, detections.shape[2]):
		# ekstrak confidence (yaitu, probabilitas) yang terkait dengan
		# deteksi
		confidence = detections[0, 0, i, 2]

		# menyaring deteksi yang lemah dengan memastikan confidence
		# lebih besar dari confidence minimum
		if confidence > args["confidence"]:
			# hitung (x, y)-koordinat kotak pembatas untuk
			# objeknya
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# pastikan kotak pembatas berada dalam dimensi
			# bingkai
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# ekstrak ROI wajah, ubah dari saluran BGR ke RGB
			# memesan, mengubah ukurannya menjadi 224x224, dan memprosesnya terlebih dahulu
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# tambahkan wajah dan kotak pembatas ke masing-masing
			# daftar
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# hanya membuat prediksi jika setidaknya satu wajah terdeteksi
	if len(faces) > 0:
		# untuk inferensi yang lebih cepat, kami akan membuat prediksi batch di *semua*
		# wajah pada saat yang sama daripada prediksi satu per satu
		# dalam perulangan `untuk` di atas
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# kembalikan 2-tupel lokasi wajah dan yang sesuai lokasi
	return (locs, preds)

# buat parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# muat model detektor wajah serial dari disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# muat model detektor masker wajah dari disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# inisialisasi aliran video dan biarkan sensor kamera siap
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames dari video stream
while True:
	# ambil frame dari video streamdan ubah ukurannya lebar maksimum 400 piksel
	frame = vs.read()
	frame = imutils.resize(frame, width=700)

	# mendeteksi wajah dalam bingkai dan menentukan apakah mereka memakai masker wajah atau tidak
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop di atas lokasi wajah yang terdeteksi dan lokasi yang sesuai
	for (box, pred) in zip(locs, preds):
		# membongkar kotak pembatas dan prediksi
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# tentukan label kelas dan warna yang akan kita gunakan untuk menggambar kotak pembatas dan teks
		if (mask > withoutMask): 
			label = "Bermasker" 
		else:
			label = "Tidak bermasker"
		color = (0, 255, 0) if label == "Bermasker" else (0, 0, 255)
			
		# sertakan probabilitas dalam label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# tampilkan label dan kotak pembatas pada bingkai keluaran
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		if mask < withoutMask:
			file = "audio1.mp3"
			os.system("" + file)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
