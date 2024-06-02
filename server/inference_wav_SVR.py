# Copyright 2022 kakaoenterprise  heize.s@kakaoenterprise.com
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# proninciation score inference
# input: wav file, output: pronunciation score

import os
import argparse
import torch
import audiofile
import joblib
from transformers import Wav2Vec2ForCTC
import numpy as np
import sys

def inference_wav(lang: str, label_type1: str, label_type2: str, dir_model: str, device: str, audio_len_max: int, wav: str):
    model_path = os.path.join(dir_model, f'lang_{lang}/svr_model_{label_type1}+{label_type2}.joblib')
    scaler_path = os.path.join(dir_model, f'lang_{lang}/scaler_{label_type1}+{label_type2}.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    # Load SVR model and scaler
    svr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if lang == 'en':
        base_model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    print(f'{lang}, {label_type1}, {label_type2}, base_model: {base_model_name}')

    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to(device)

    x, sr = audiofile.read(wav)
    x = torch.tensor(x[:min(x.shape[-1], audio_len_max)], device=device).reshape(1, -1)
    feat_x = base_model(x, output_attentions=True, output_hidden_states=True, return_dict=True).hidden_states[-1]
    feat_x = torch.mean(feat_x, axis=1).cpu().detach().numpy()

    # Scale features
    feat_x = scaler.transform(feat_x)

    # Predict pronunciation score using SVR
    pred_score = svr_model.predict(feat_x)
    pred_score = np.clip(pred_score, 0, 5)

    print(f'score: {pred_score[0]}')
    return pred_score[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str)
    parser.add_argument("--label_type1", default='pron', type=str, help='fluency, pron')
    parser.add_argument("--label_type2", default='prosody', type=str, help='articulation, prosody')
    parser.add_argument("--dir_model", default='model_svr_ckpt', type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--audio_len_max", default=200000, type=int)
    parser.add_argument("--wav", type=str, required=True)

    args = parser.parse_args()

    if args.lang == 'en':
        args.base_model = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    inference_wav(args.lang, args.label_type1, args.label_type2, args.dir_model, args.device, args.audio_len_max, args.wav)

#python3 inference_wav_SVR.py --lang en --label_type1 pron --label_type2 prosody --dir_model model_svr_ckpt --wav ./example_TTS.wav