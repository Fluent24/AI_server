import os
import torch
import audiofile
from transformers import Wav2Vec2ForCTC
import numpy as np
import sys
sys.path.insert(0, './server')
from score_model import MLP

def inference_wav2(lang: str, label_type1: str, label_type2: str, dir_model: str, device: str, audio_len_max: int, wav: str):
    model_path = os.path.join(dir_model, f'lang_{lang}', f'{label_type1}_{label_type2}_checkpoint.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    score_model = torch.load(model_path, map_location=device)
    score_model.eval()

    base_model_name = {
        'en': 'facebook/wav2vec2-large-robust-ft-libri-960h',
        'jp': 'NTQAI/wav2vec2-large-japanese',
        'zh': 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
        'de': 'facebook/wav2vec2-large-xlsr-53-german',
        'es': 'facebook/wav2vec2-large-xlsr-53-spanish',
        'fr': 'facebook/wav2vec2-large-xlsr-53-french',
        'ru': 'bond005/wav2vec2-large-ru-golos'
    }.get(lang, 'facebook/wav2vec2-large-robust-ft-libri-960h')

    print(f'{lang}, {label_type1}, {label_type2}, base_model: {base_model_name}')

    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to(device)

    x, sr = audiofile.read(wav)
    x = torch.tensor(x[:min(x.shape[-1], audio_len_max)], dtype=torch.float32).reshape(1, -1).to(device)
    
    with torch.no_grad():
        feat_x = base_model(x, output_attentions=False, output_hidden_states=True, return_dict=True).hidden_states[-1]
        feat_x = torch.mean(feat_x, axis=1).to(device)

    score_model.to(device)
    pred_score = score_model(feat_x).cpu().detach().numpy()
    pred_score = np.clip(pred_score, 0, 5)

    print(f'score: {pred_score[0][0]}')
    
    # Clear GPU memory
    del x, feat_x, base_model, score_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_score[0][0]