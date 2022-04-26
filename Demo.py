from typing import Generator
import matplotlib

import matplotlib.pyplot as plt
from SimuClasses import *
from tkinter import *
import time
import os
from tkinter import messagebox
from PIL import ImageTk, Image
import sys
import tkinter.font
import requests
from tkinter.filedialog import askopenfilename, askdirectory
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FONT = "Verdana"

class DemoGUI:
    def split(self):

        return "                                                                                                       "    

    def startUI(self):

        def set_text(entry, text):
            entry.delete(0, END)
            entry.insert(0, text)
            return None

        def set_label_text(label, text):
            label.config(text=text)
            return None

        def refresh(self):
            self.destroy()
            self.__init__()
            
        def Giris():
            def message(text):
                set_label_text(result,text)
            # 2 Entry, 3 Button, 1 checkbutton,1 Label olacak
            # simulasyon butonuna basıldığında 2 pencere daha çıkacak biri map'ı gösterecek diğeri de step monitor
            mainGUI = Tk()
            boyut = 800
            mainGUI.geometry("%dx%d" % (boyut, boyut))
            mainGUI.title("UAV Simulation Demo")
            mainGUI.resizable(False, False)

            
            result = None
            ModelFolder = None
            UserData = None

            UserDataDialog = None
            ModelFolderDialog = None

            generateRandomEnv = None
            generateOptionCheckBox = None
            StartSimuButton = None

            def load_file():
                fname = fname = askopenfilename(defaultextension='.pkl')
                set_text(UserData, fname)

            def load_dir():
                fname = askdirectory()
                set_text(ModelFolder, fname)

            def simulation_start():
                result_dict = {}
                gen = test_env(ModelFolder.get(),UserData.get(),
                        random_env = generateRandomEnv.get(),step_count=int(StepCountEntry.get()),
                        plot_trajectories=showTrajectoriesVar.get())
                for i in gen: 
                    plt.pause(0.0000000000001)
                    result_dict = i
                set_label_text(result,"Average Sum Rate Score: {:.2f}\nFinal Sum Rate Score: {:.2f}"
                .format(*(result_dict["scores"])))
                
            def disable_dialog():
                def switch(obj):
                    st = obj["state"]
                    obj["state"] = "normal" if st=="disabled" else "disabled"

                switch(UserData)
                switch(UserDataDialog)

            def render():
                # Sol üst 0,0 yani Aşağı gidildikçe y artıyor
                def LOC(X,Y,obj):
                    return {"relx":X,"rely":Y,"obj":obj}

                COL_START_X = 0.2
                COL_SPACE_X = 0.4
                COL2_START_X = COL_START_X + COL_SPACE_X

                resultLoc = LOC(0.35,0.1,result)

                ModelFolderLoc,ModelFolderDialogLoc = LOC(COL_START_X, 0.4,ModelFolder),LOC(COL2_START_X, 0.39,ModelFolderDialog)

                generateOptionCheckBoxLoc = LOC(COL_START_X,0.5,generateOptionCheckBox)

                UserDataLoc,UserDataDialogLoc = LOC(COL_START_X, 0.6,UserData),LOC(COL2_START_X, 0.59,UserDataDialog)

                stepCountLabelLoc, StepCountEntryLoc = LOC(COL_START_X, 0.7,stepCountLabel),LOC(COL_START_X+0.2, 0.7,StepCountEntry)

                showTrajectoriesCheckBoxLoc = LOC(COL_START_X, 0.8,showTrajectoriesCheckBox)

                StartSimuButtonLoc = LOC(0.35,0.9,StartSimuButton)

                
                pack = [resultLoc,
                        generateOptionCheckBoxLoc,
                        UserDataLoc,UserDataDialogLoc,
                        ModelFolderLoc,ModelFolderDialogLoc,
                        stepCountLabelLoc,StepCountEntryLoc,
                        showTrajectoriesCheckBoxLoc,
                        StartSimuButtonLoc]

                for location in pack:
                    location.pop("obj").place(**location)
                


            result = Label(mainGUI, bg="white", fg="Black",
                           font=FONT+ " 13", text="")
            ModelFolder = Entry(
                mainGUI, relief=FLAT, bg="white", fg="black",
                font=FONT+" 15 italic", text="")
            UserData = Entry(
                mainGUI, relief=FLAT, bg="white", fg="black",
                font=FONT+" 15 italic", text="")

            UserDataDialog = Button(mainGUI, relief=FLAT, bg="white", fg="black",
                font=FONT+" 15 italic", text="Choose User Data",
                                command=load_file)
            ModelFolderDialog = Button(mainGUI, relief=FLAT, bg="white", fg="black",
                font=FONT+" 15 italic", text="Choose Model Folder",
                                command=load_dir)

            generateRandomEnv = tkinter.BooleanVar()
            generateOptionCheckBox = tkinter.Checkbutton(mainGUI, text='Random Environment',variable=generateRandomEnv, 
            font=FONT+ " 15 italic", onvalue=1, offvalue=0, command=disable_dialog)

            stepCountLabel = Label(mainGUI, fg="Black",
                           font=FONT+ " 13", text="Step Count:")

            StepCountEntry = Entry(
                mainGUI, relief=FLAT, bg="white", fg="black",
                font=FONT+" 15 italic", text="64")

            showTrajectoriesVar = tkinter.BooleanVar()
            showTrajectoriesCheckBox = tkinter.Checkbutton(mainGUI, text='Show Past Trajectories',variable=showTrajectoriesVar, 
            font=FONT+ " 15 italic", onvalue=1, offvalue=0)

            StartSimuButton = Button(mainGUI, relief=FLAT, bg="white", fg="black", font=FONT+ " 15 italic", text="Start Simulation",
                          command=simulation_start)

            render()

        Giris()
        mainloop()


gui = DemoGUI()
gui.startUI()
