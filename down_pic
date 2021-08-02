from os import write
import numpy as np
import pandas as pd
import requests
import cv2

data=pd.read_csv("E:/Code/fer_project\FEC_dataset/faceexp-comparison-data-train-public.csv",header=None,error_bad_lines=False)
path="E:/Code/fer_project/FEC_dataset/picture_fer/"
for i in range(3000):
    try:
        num=0
        url=data.iloc[i,num*5]
        name=path+data.iloc[i,num*5].split('/')[-1]
        reponse=requests.get(url)
        file = open(name,"wb")
        file.write(reponse.content)
        file.close()
        reponse.raise_for_status()

        img=cv2.imread(name)
        x=img.shape
        print(name)
        img=img[int(data.iloc[i,num*5+3]*x[0]):int(data.iloc[i,num*5+4]*x[0]),int(data.iloc[i,num*5+1]*x[1]):int(data.iloc[i,num*5+2]*x[1])]
        save_img=cv2.resize(img,(256,256))
        cv2.imwrite(name,save_img)
    except requests.exceptions.RequestException: 
        continue
