# dataset.py (수정된 버전)

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import os

class PMEmoDataset(Dataset):
    def __init__(self, pmemo_root_path):
        self.audio_dir = os.path.join(pmemo_root_path, 'chorus')
        
        metadata_path = os.path.join(pmemo_root_path, 'metadata.csv')
        annotations_path = os.path.join(pmemo_root_path, 'annotations/dynamic_annotations.csv')
        
        self.metadata = pd.read_csv(metadata_path)
        self.annotations = pd.read_csv(annotations_path)
        
        self.song_ids = self.annotations['song_id'].unique()

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        song_id = self.song_ids[idx]
        
        # 1. Search Audio File Path
        audio_filename = f"{song_id}.mp3"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        # 2. Load Corresponding Song's Dynamic Annotation
        song_annotations = self.annotations[self.annotations['song_id'] == song_id]
        labels = song_annotations[['valence', 'arousal']].values.astype(np.float32)
        
        # 3. Load Audio File
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 4. Extract Feature Sequence
        frame_length = int(0.5 * sr)
        num_frames = len(labels)
        
        feature_sequence = []
        for i in range(num_frames):
            start_sample = i * frame_length
            end_sample = start_sample + frame_length
            
            # 프레임이 오디오 길이를 벗어나지 않도록 처리
            if end_sample > len(y):
                frame = y[start_sample:]
            else:
                frame = y[start_sample:end_sample]
            
            # 현재 프레임에서 MFCC 특징 추출
            mfccs = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            feature_sequence.append(mfccs_mean)
            
        return {
            "features": torch.tensor(np.array(feature_sequence), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32)
        }