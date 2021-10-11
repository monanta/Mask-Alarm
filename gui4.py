# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import os
import threading


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # buka sumber video (secara default ini akan mencoba membuka webcam komputer)
        self.vid = MyVideoCapture(self.video_source)

        # Buat kanvas yang sesuai dengan ukuran sumber video di atas
        self.window.resizable(False, False)
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()  

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
        time.sleep(2.0)

        # Tombol yang memungkinkan pengguna mengambil snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.memotret)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_snapshot=tkinter.Button(window, text="Mulai", width=50, command=self.mulai)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # threading.Timer(2, self.update(faceNet, maskNet)).start()

        # Setelah dipanggil sekali, metode pembaruan akan secara otomatis dipanggil setiap milidetik penundaan
        # self.delay = 1500
        # self.update(faceNet, maskNet)
        self.delay = 15
        self.window.mainloop()
    
    def mulai(self):
        ret, frame = self.vid.get_frame()
    
        if ret:
            frame = imutils.resize(frame, width=700)
            # (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        


    def memotret(self):
        # Memotret orang yang tidak memakai masker
        ret, frame = self.vid.get_frame()
        if ret:
            path = './tertangkap'
            namafile = "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
            cv2.imwrite(os.path.join(path , namafile), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

            

    def update(self, faceNet, maskNet):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # -------------------
            frame = imutils.resize(frame, width=700)

            # mendeteksi wajah dalam bingkai dan menentukan apakah mereka memakai masker wajah atau tidak
            (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

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
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF

            
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        # self.window.after(self.delay, self.update)
    
    def detect_and_predict_mask(self, frame, faceNet, maskNet):
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


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()



# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")