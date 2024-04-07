import os
from os.path import basename, splitext, join as path_join
import re
import json
from glob import glob
from moviepy.editor import VideoFileClip

LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'

# some .avi video files are cut too early, leaving visual scenes for these wav files inaccessible
missing = [
    "Session1/sentences/wav/Ses01F_impro04/Ses01F_impro04_M036.wav",
    "Session1/sentences/wav/Ses01F_script02_1/Ses01F_script02_1_M044.wav",
    "Session1/sentences/wav/Ses01F_script03_1/Ses01F_script03_1_F031.wav",
    "Session1/sentences/wav/Ses01M_impro02/Ses01M_impro02_F021.wav",
    "Session1/sentences/wav/Ses01M_impro04/Ses01M_impro04_M025.wav",
    "Session1/sentences/wav/Ses01M_script02_1/Ses01M_script02_1_F024.wav",
    "Session2/sentences/wav/Ses02F_script01_1/Ses02F_script01_1_F044.wav",
    "Session2/sentences/wav/Ses02F_script01_2/Ses02F_script01_2_M018.wav",
    "Session2/sentences/wav/Ses02F_script02_2/Ses02F_script02_2_M046.wav",
    "Session2/sentences/wav/Ses02M_script02_2/Ses02M_script02_2_F046.wav",
    "Session3/sentences/wav/Ses03F_impro01/Ses03F_impro01_F011.wav",
    "Session3/sentences/wav/Ses03F_impro01/Ses03F_impro01_M011.wav",
    "Session3/sentences/wav/Ses03F_impro06/Ses03F_impro06_FXX1.wav",
    "Session3/sentences/wav/Ses03M_impro04/Ses03M_impro04_M041.wav",
    "Session3/sentences/wav/Ses03M_impro07/Ses03M_impro07_F023.wav",
    "Session3/sentences/wav/Ses03M_impro07/Ses03M_impro07_M026.wav",
    "Session4/sentences/wav/Ses04F_impro05/Ses04F_impro05_M024.wav",
    "Session4/sentences/wav/Ses04M_script02_1/Ses04M_script02_1_F021.wav",
]

def get_wav_paths(data_dirs):
    wav_paths = glob(path_join(data_dirs, '**/*.wav'), recursive=True)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        if wav_path not in missing:
            wav_dict[wav_name] = wav_path

    return wav_dict

def write_metadata_json(data_dirs, paths, out_path):
    meta_data = []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt' and not label_path.startswith('._')]
        
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
    if not os.path.exists(clip_filename):
        video_clip = VideoFileClip(video_filename).subclip(start_time, end_time)
        video_clip.write_videofile(clip_filename, codec = "libx264", logger=None)

def avi_to_clips(data_dir, clips_output_dir, i, path):
    avi_path = path_join(data_dir, path, '/dialog/avi/DivX/')
    avi_all = []
    for root, dirs, files in os.walk(avi_path):
        for file in files:
            if file.endswith('.avi') and not file.startswith('._'):
                avi_all.append(path_join(avi_path, file))

    for avi in avi_all:
        video_filename = avi
        raw_name = os.path.splitext(avi.split('/')[-1])[0]
        lab_F = path_join(data_dir, path, '/dialog/lab/Ses0', str(i+1), '_F/', raw_name, '.lab')
        lab_M = path_join(data_dir, path, '/dialog/lab/Ses0', str(i+1), '_M/', raw_name, '.lab')
        
        clip_dir = path_join(clips_output_dir, path, raw_name)
        os.makedirs(clip_dir, exist_ok=True)

        f_F = open(lab_F, 'r')
        sentences_F = [line.strip() for line in f_F.readlines()[1:]]
        f_F.close()

        f_M = open(lab_M, 'r')
        sentences_M = [line.strip() for line in f_M.readlines()[1:]]
        f_M.close()

        for sen_F in sentences_F:
            if sen_F.split(' ')[-1] not in [f.split('/')[-1].split('.')[0] for f in missing]:
                clip_filename = clip_dir+'/'+sen_F.split(' ')[-1]+'.mp4'
                try:
                    video_clip_store(video_filename, clip_filename, sen_F.split(' ')[0], sen_F.split(' ')[1]) 
                except OSError as e:
                    print(f"Video: {avi}")
                    print(f"Label: {lab_F}")
                    print(f"Line: {sen_F}")
                    print(sen_F.split(' ')[-1])
                    print(e)

        for sen_M in sentences_M:
            if sen_M.split(' ')[-1] not in [f.split('/')[-1].split('.')[0] for f in missing]:
                clip_filename = clip_dir+'/'+sen_M.split(' ')[-1]+'.mp4'
                try:
                    video_clip_store(video_filename, clip_filename, sen_M.split(' ')[0], sen_M.split(' ')[1]) 
                except OSError as e:
                    print(f"Video: {avi}")
                    print(f"Label: {lab_M}")
                    print(f"Line: {sen_M}")
                    print(sen_M.split(' ')[-1])
                    print(e)
          
def main(data_dir, clips_output_dir, metadata_output_dir):
    """Main function."""
    paths = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

    for i, path in enumerate(paths):
        os.makedirs(f"{metadata_output_dir}/{path}", exist_ok=True)

        train_metadata = path_join(f"{metadata_output_dir}/{path}", 'train_meta_data.json')
        write_metadata_json(data_dir, [p for p in paths if p != path], train_metadata)

        test_metadata = path_join(f"{metadata_output_dir}/{path}", 'test_meta_data.json')
        write_metadata_json(data_dir, [path], test_metadata)

        avi_to_clips(data_dir, clips_output_dir, i, path)

if __name__ == "__main__":
    data_dir = "/media/rogert/DATA1/IEMOCAP_full_release"
    assert os.path.exists(data_dir), f"Make sure IEMOCAP exists at {data_dir}."

    clips_output_dir = path_join(data_dir, 'clips')
    metadata_output_dir = path_join(data_dir, 'meta_data')

    main(data_dir, clips_output_dir, metadata_output_dir)
