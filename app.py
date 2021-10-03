from io import DEFAULT_BUFFER_SIZE
import tkinter as tk
from tkinter import Frame, filedialog, Text
import os

root = tk.Tk()

canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")
canvas.pack()

Frame = tk.Frame(root, bg="white")
Frame.place(relwidth=0.8, relheigh=0.8, relx=0.1, rely=0.1)

openFile = tk.Button(root, text="Open File", padx=10, 
                    pady=5, fg="white", bg="#263D42")
openFile.pack()

root.mainloop()