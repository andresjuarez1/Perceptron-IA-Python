import tkinter as tk
from tkinter import filedialog, ttk
import main as per
import threading
import numpy as np

def selectFile():
    filename = filedialog.askopenfilename(initialdir="/", title="Selecciona el archivo", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
    fileLabel.config(text=filename)
    return filename



def startTrain():
    try:
        tasaDeAprendizaje = float(tasaDeAprendizajeInput.get())
        epocas = int(epocasInput.get())
        rutaArchivo = fileLabel.cget("text")
        threading.Thread(target=lambda: per.trainPerceptron(tasaDeAprendizaje, epocas, rutaArchivo)).start()
        np.set_printoptions(precision=4, suppress=True)

        per.showResults()
    except ValueError:
        print("Ingresa valores numéricos")
    

root = tk.Tk()

root.title("PERCEPTON ENTRENAMIENTO BY ANDRES JUAREZ")
root.geometry('700x400')

style = ttk.Style()
style.theme_use('clam')

frame = ttk.Frame(root, padding="10 10 10 10")
frame.pack(fill=tk.BOTH, expand=True)

style.configure("TEntry", padding=(5, 5, 5, 5))



ttk.Label(frame, text="Épocas:").pack(pady=5)
epocasInput = ttk.Entry(frame, width=30,  style="TEntry")
epocasInput.pack(pady=10) 

ttk.Label(frame, text="Tasa de aprendizaje:").pack(pady=5)
tasaDeAprendizajeInput = ttk.Entry(frame, width=30,  style="TEntry")  
tasaDeAprendizajeInput.pack(pady=10)


ttk.Button(frame, text="Selecciona un CSV", command=selectFile).pack(pady=10)
fileLabel = ttk.Label(frame, text="")

ttk.Button(frame, text="START", command=startTrain).pack(pady=10)


root.mainloop()