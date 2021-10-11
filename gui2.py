import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import os


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

        # Tombol yang memungkinkan pengguna mengambil snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.memotret)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # Setelah dipanggil sekali, metode pembaruan akan secara otomatis dipanggil setiap milidetik penundaan
        self.delay = 15
        self.update()

        self.window.mainloop()
    

    def memotret(self):
        # Memotret orang yang tidak memakai masker
        ret, frame = self.vid.get_frame()
        if ret:
            path = './tertangkap'
            namafile = "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
            cv2.imwrite(os.path.join(path , namafile), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

            

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


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