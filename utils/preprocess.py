import warnings

def preprocess(model, frames, wav, video_fps, audio_sr, device):
    if hasattr(model, 'preprocess') and callable(model.preprocess):
        # for AVBERT
        processed_frames, processed_wav = model.preprocess(frames, wav, video_fps, audio_sr)
    else:
        if hasattr(model, 'preprocess_audio') and callable(model.preprocess_audio):
            processed_wav = model.preprocess_audio(wav, audio_sr)
        else:
            warnings.warn('Model does not implement preprocess_audio method.')
            processed_wav = wav
        if hasattr(model, 'preprocess_video') and callable(model.preprocess_video):
            processed_frames = model.preprocess_video(frames, video_fps)
        else:
            warnings.warn('Model does not implement preprocess_video method.')
            processed_frames = frames
    return processed_wav.to(device), processed_frames.to(device),