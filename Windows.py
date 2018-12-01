# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:58:11 2018

@author: Vincent
"""
import os
import time
import tkinter
from tkinter import *
import tkinter.filedialog
from PIL import Image,ImageTk
import cv2 as cv
import os.path
import glob
from multiprocessing import Process
from multiprocessing import Pool
import threading
from object_detection_yolo_offline import *

root = Tk()
root.title('车辆检测')

root.resizable(False, False)

def chooseFile():
    filename = tkinter.filedialog.askopenfilename()
    if filename!= '':
        videoname=filename.split('/')[-1]
        frames_dir = videoname.split('.')[0]
        thread1=threading.Thread(target=video2imgs,name='video2imgs',kwargs={'video_name':filename,'frames_dir':frames_dir},daemon=True)
        thread1.start()
        thread2=threading.Thread(target=detect,name='detect',kwargs={'video':filename},daemon=True)
        thread2.start()
        time.sleep(8)
        thread3=threading.Thread(target=Play,name='play',kwargs={'filePath':'output_frames'},daemon=True)
        thread3.start()
    else:
        lb.config(text = "您没有选择任何文件");

def selectJpg(jpgfile,width=960,height=540):
    img=Image.open(jpgfile)
    imgrz=img.resize((width,height),Image.ANTIALIAS)
    imgTK=ImageTk.PhotoImage(imgrz)
    return imgTK

def changeimg(img):
    lb.configure(image=img,text='')

def Play(filePath,idx=0):
    imgList=os.listdir(filePath)
    global imgplay
    listlen=len(imgList)
    if idx<listlen:
      start=time.time()
      imgname = filePath+'/core-{:02d}.jpg'.format(idx)
      idx+=1
      imgplay=selectJpg(imgname)
      lb.configure(image=imgplay)
      lb.after(20,Play,filePath,idx)
      
    else:
        return
if __name__=='__main__':
    
    #封面
    cover=selectJpg('source/cover.jpg',1000,640)
    lb=Label(root,image=cover,fg='black')
    lb.pack(fill=X)
    init=selectJpg('source/init.jpg')
    lb.after(3000,changeimg,init)
    
    #选择文件
    btn = Button(root,text="选择视频文件",command=chooseFile,padx=30,pady=10)
    btn.pack()
    quitbtn=Button(root,text='退出',command=root.quit,padx=53,pady=10)
    quitbtn.pack()
    root.geometry('1000x640')
    root.deiconify
    root.mainloop()
