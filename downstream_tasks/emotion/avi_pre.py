import numpy as np
import os
import imageio
import cv2
from moviepy.editor import*
import shutil
import time
from joblib import Parallel, delayed
import sys

def video_clip_npy_store(i, video_filename, npy_filename, start_time, end_time):
    video_clip = VideoFileClip(video_filename).subclip(start_time, end_time)
    video_clip.write_videofile('./IEMOCAP/Session'+str(i)+'/sentences/avi_sentence/tmp.mp4', codec = "libx264")
    clip_filename = './IEMOCAP/Session'+str(i)+'/sentences/avi_sentence/tmp.mp4'
    vid = imageio.get_reader(clip_filename, 'ffmpeg')
    capture = cv2.VideoCapture(clip_filename)
    frame_count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
    images_np = []

    for frame_num in range(int(frame_count)):
        image = vid.get_data(frame_num)
        npimage = np.array(image)
        images_np.append(npimage)

    npimage_stack = np.stack(images_np)
    np.save(npy_filename, npimage_stack)

def avi_pre(i):
    path = './IEMOCAP/Session'+str(i)+'/dialog/avi/DivX/'
    avi_all = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.avi'):
                avi_all.append(os.path.join(path, file))

    for avi in avi_all:
        video_filename = avi
        raw_name = os.path.splitext(avi.split('/')[-1])[0]
        lab_F = '/data/member1/user_tahsieh/IEMOCAP/Session'+str(i)+'/dialog/lab/Ses01_F/'+raw_name+'.lab'
        lab_M = '/data/member1/user_tahsieh/IEMOCAP/Session'+str(i)+'/dialog/lab/Ses01_M/'+raw_name+'.lab'
        npy_dir = './IEMOCAP/Session'+str(i)+'/sentences/avi_npy/'+raw_name
        if not os.path.isdir(npy_dir):
                os.mkdir(npy_dir)

        f_F = open(lab_F, 'r')
        line_F = f_F.readline()
        sentences_F = []
        while line_F:
            line_F = f_F.readline().replace('\n','') 
            sentences_F.append(line_F)
        f_F.close()

        f_M = open(lab_M, 'r')
        line_M = f_M.readline()
        sentences_M = []
        while line_M:
            line_M = f_M.readline().replace('\n','') 
            sentences_M.append(line_M)
        f_M.close()

        for sen_F in sentences_F:
            npy_filename = npy_dir+'/'+sen_F.split(' ')[-1]+'.npy'
            try:
                video_clip_npy_store(i, video_filename, npy_filename, sen_F.split(' ')[0], sen_F.split(' ')[1])
            except:
                f = open('./IEMOCAP/miss.txt', 'w')
                f.write('Session'+str(i)+':')
                f.write(file)
                f.write("\n")
                f.close()
                continue
        for sen_M in sentences_M:
            npy_filename = npy_dir+'/'+sen_M.split(' ')[-1]+'.npy'
            try:
                video_clip_npy_store(i, video_filename, npy_filename, sen_M.split(' ')[0], sen_M.split(' ')[1]) 
            except:
                f = open('./IEMOCAP/miss.txt', 'w')
                f.write('Session'+str(i)+':')
                f.write(file)
                f.write("\n")
                f.close()
                continue
                
%%time
ret = Parallel(n_jobs=16)(delayed(avi_pre)(i) for i in range(1,6))