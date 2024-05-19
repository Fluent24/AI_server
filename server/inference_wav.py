# Copyright 2022 kakaoenterprise  heize.s@kakaoenterprise.com
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# proninciation score inference
# input: wav file, output: pronunciation score

import os, argparse
import torch
import audiofile
from transformers import Wav2Vec2ForCTC
import numpy as np
import sys
sys.path.insert(0, './server')
from score_model import MLP

def inference_wav(lang: str, label_type1: str, label_type2: str, dir_model: str, device: str, audio_len_max: int, wav: str):
    dir_model = os.path.join(dir_model, f'lang_{lang}', f'{label_type1}_{label_type2}_checkpoint.pt')
    
    if not os.path.exists(dir_model):
        raise FileNotFoundError(f"Model checkpoint not found at {dir_model}")
    
    score_model = torch.load(dir_model, map_location=device)
    score_model.eval()

    if lang == 'en':
        base_model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'
    elif lang == 'jp':
        base_model_name = 'NTQAI/wav2vec2-large-japanese'
    elif lang == 'zh':
        base_model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn'
    elif lang == 'de':
        base_model_name = 'facebook/wav2vec2-large-xlsr-53-german'
    elif lang == 'es':
        base_model_name = 'facebook/wav2vec2-large-xlsr-53-spanish'
    elif lang == 'fr':
        base_model_name = 'facebook/wav2vec2-large-xlsr-53-french'
    elif lang == 'ru':
        base_model_name = 'bond005/wav2vec2-large-ru-golos'

    print(f'{lang}, {label_type1}, {label_type2}, base_model: {base_model_name}')

    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to(device)

    x, sr = audiofile.read(wav)
    x = torch.tensor(x[:min(x.shape[-1], audio_len_max)], device=device).reshape(1, -1)
    feat_x = base_model(x, output_attentions=True, output_hidden_states=True, return_dict=True).hidden_states[-1]
    feat_x = torch.mean(feat_x, axis=1)

    pred_score = score_model(feat_x).cpu().detach().numpy()
    pred_score = np.clip(pred_score, 0, 5)

    print(f'score: {pred_score[0][0]}')
    return pred_score[0][0]