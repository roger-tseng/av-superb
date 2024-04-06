import os
from os.path import basename, splitext, join as path_join
import sys
import re
import json
from glob import glob
import shutil
from moviepy.editor import*


LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'
DATA_DIR = '/media/rogert/DATA1/IEMOCAP_full_release/'
CLIP_DIR = '/media/rogert/DATA1/IEMOCAP_full_release/clips/'

miss = 'downstream_tasks/emotion/miss.txt' 
miss_list = []
miss_filename = []
with open(miss, 'r') as f:
    line = f.readline()
    while line:
        line = f.readline().replace('\n','') #去掉換行
        miss_list.append(line)
        miss_filename.append((os.path.splitext(line)[0]).split('/')[-1])  
f.close()



def get_wav_paths(data_dirs):
    wav_paths = glob(os.path.join(data_dirs, '**/*.wav'), recursive=True)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        wav_dict[wav_name] = wav_path

    return wav_dict


def preprocess(data_dirs, paths, out_path):
    meta_data = []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            with open(path_join(label_dir, label_path)) as f:
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['neu', 'hap', 'ang', 'sad', 'exc']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    meta_data.append({
                        'path': wav_paths[line[1]],
                        'label': line[2].replace('exc', 'hap'),
                        'speaker': re.split('_', basename(wav_paths[line[1]]))[0]
                    })
    data = {
        'labels': {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3},
        'meta_data': meta_data
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)
        
def video_clip_store(video_filename, clip_filename, start_time, end_time):
    video_clip = VideoFileClip(video_filename).subclip(start_time, end_time)
    video_clip.write_videofile(clip_filename, codec = "libx264")

def avi_preprocess(i, path):
    avi_path = DATA_DIR+path+'/dialog/avi/DivX/'
    avi_all = []
    for root, dirs, files in os.walk(avi_path):
        for file in files:
            if file.endswith('.avi'):
                avi_all.append(os.path.join(avi_path, file))

    for avi in avi_all:
        video_filename = avi
        raw_name = os.path.splitext(avi.split('/')[-1])[0]
        lab_F = DATA_DIR+path+'/dialog/lab/Ses0'+str(i+1)+'_F/'+raw_name+'.lab'
        lab_M = DATA_DIR+path+'/dialog/lab/Ses0'+str(i+1)+'_M/'+raw_name+'.lab'
        clip_dir = os.path.join(CLIP_DIR, path, raw_name)
        if not os.path.isdir(clip_dir):
            os.makedirs(clip_dir)

        f_F = open(lab_F, 'r', encoding='iso-8859-1')
        line_F = f_F.readline()
        sentences_F = []
        while line_F:
            line_F = f_F.readline().replace('\n','')
            sentences_F.append(line_F)
        f_F.close()

        f_M = open(lab_M, 'r', encoding='iso-8859-1')
        line_M = f_M.readline()
        sentences_M = []
        while line_M:
            line_M = f_M.readline().replace('\n','')
            sentences_M.append(line_M)
        f_M.close()

        for sen_F in sentences_F:
            if sen_F not in miss_filename:
                clip_filename = clip_dir+'/'+sen_F.split(' ')[-1]+'.mp4'
                video_clip_store(video_filename, clip_filename, sen_F.split(' ')[0], sen_F.split(' ')[1]) 
        for sen_M in sentences_M:
            if sen_M not in miss_filename:
                clip_filename = clip_dir+'/'+sen_M.split(' ')[-1]+'.mp4'
                video_clip_store(video_filename, clip_filename, sen_M.split(' ')[0], sen_M.split(' ')[1]) 
          
def main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    for i, path in enumerate(paths):
        os.makedirs(f"{out_dir}/{path}", exist_ok=True)
        preprocess(data_dir, paths[:i] + paths[i + 1:], path_join(f"{out_dir}/{path}", 'train_meta_data.json'))
        preprocess(data_dir, [path], path_join(f"{out_dir}/{path}", 'test_meta_data.json'))
        avi_preprocess(i, path)

if __name__ == "__main__":
    main("/media/rogert/DATA/IEMOCAP_full_release")