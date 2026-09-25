"""
词级强制对齐脚本（whisperx）
输入：100 条样本 pkl（位于 data/processed/features_q1/single/）
      对应原始视频（位于 data/raw_attachment1/MOSEI数据集部分原始视频-100条/）
输出：每条样本一个 alignment JSON（位于 data/processed/features_q1/alignment/）

对齐逻辑：
1. 从视频抽音频（ffmpeg）
2. whisperx ASR 转写 → 拿到词级时间戳 + ASR 文本
3. ASR 文本 vs raw_text：计算 WER
4. 把 raw_text 的 word 与 ASR word 对齐（DP 匹配）
5. 每个 raw_word 对应 [audio_start, audio_end] + [video_frame_start, video_frame_end]
"""
import os
import sys
import json
import argparse
import pickle
import subprocess
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
DEVICE = 'cuda' if (torch.cuda.is_available() and os.environ.get('CUDA_FORCE') == '1') else 'cpu'
COMPUTE_TYPE = 'int8' if DEVICE == 'cpu' else 'float16'


def extract_audio_from_video(video_path: str, out_wav: str) -> bool:
    """用 ffmpeg 从视频抽音频为 16k 单声道 wav"""
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ac', '1', '-ar', '16000',
        '-loglevel', 'error',
        out_wav
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60)
        return os.path.exists(out_wav)
    except Exception as e:
        print(f"  ffmpeg 抽音频失败: {e}")
        return False


def get_video_fps(video_path: str) -> float:
    """从视频读取 fps"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate,nb_frames,duration',
        '-of', 'json', video_path
    ]
    try:
        out = subprocess.check_output(cmd, timeout=30).decode()
        info = json.loads(out)
        stream = info['streams'][0]
        if 'nb_frames' in stream and 'duration' in stream:
            n = float(stream['nb_frames'])
            d = float(stream['duration'])
            if d > 0:
                return n / d
        r = stream.get('r_frame_rate', '30/1')
        if '/' in r:
            num, den = r.split('/')
            if float(den) > 0:
                return float(num) / float(den)
        return 30.0
    except Exception:
        return 30.0


def get_audio_duration(wav_path: str) -> float:
    """从 wav 文件读取时长"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json', wav_path
    ]
    try:
        out = subprocess.check_output(cmd, timeout=10).decode()
        return float(json.loads(out)['format']['duration'])
    except Exception:
        return 0.0


def compute_wer(ref: str, hyp: str) -> float:
    """计算 WER（ref=ground-truth, hyp=ASR 输出）"""
    try:
        from jiwer import wer
        return round(wer(ref, hyp), 4)
    except ImportError:
        ref_words = ref.split()
        hyp_words = hyp.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        n = len(ref_words)
        m = len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i-1].lower() == hyp_words[j-1].lower() else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        return round(dp[n][m] / max(n, 1), 4)


def load_whisperx_models(model_size: str = 'base'):
    """加载 whisperx 模型（ASR + 对齐模型）"""
    import whisperx
    print(f"  加载 whisperx ASR 模型 ({model_size})...")
    asr_model = whisperx.load_model(model_size, DEVICE, compute_type=COMPUTE_TYPE)
    print(f"  加载对齐模型 (wav2vec2-base-960h)...")
    align_model, metadata = whisperx.load_align_model(language_code='en', device=DEVICE)
    return asr_model, align_model, metadata


def _norm(w: str) -> str:
    return ''.join(c for c in w.lower() if c.isalnum())


def _dp_match_words(raw_words: List[str], asr_words: List[str]) -> Dict[int, List[int]]:
    """DP 匹配 raw_words 到 asr_words，返回 {raw_idx: [asr_idx,...]}"""
    n, m = len(raw_words), len(asr_words)
    if n == 0 or m == 0:
        return {}

    raw_norm = [_norm(w) for w in raw_words]
    asr_norm = [_norm(w) for w in asr_words]

    INF = 1e9
    dp = [[INF] * m for _ in range(n)]
    dp[0][0] = 0 if raw_norm[0] == asr_norm[0] else 1

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates = []
            if raw_norm[i] == asr_norm[j]:
                if j > 0:
                    candidates.append(min(dp[i-1][k] for k in range(j)))
                else:
                    candidates.append(dp[i-1][0])
            if j > 0:
                candidates.append(dp[i][j-1] + 1)
            if i > 0:
                candidates.append(dp[i-1][j] + 1)
            dp[i][j] = min(candidates)

    match = {i: [] for i in range(n)}
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        if raw_norm[i] == asr_norm[j]:
            best_k = j
            if j > 0:
                best_k = min(range(j), key=lambda k: dp[i-1][k])
            match[i].append(j)
            i -= 1
            if i >= 0:
                j = best_k
            else:
                break
        else:
            if i > 0 and dp[i-1][j] <= dp[i][j-1] + 1:
                i -= 1
            else:
                j -= 1
    for k in match:
        match[k].sort()
    return match


def align_one_sample(sample_pkl_path: str,
                     video_base_dir: str,
                     asr_model,
                     align_model,
                     align_metadata,
                     out_json_path: str,
                     max_seq_len: int = 50) -> Dict:
    """对齐单条样本"""
    with open(sample_pkl_path, 'rb') as f:
        sample = pickle.load(f)

    video_id = sample['video_id']
    clip_id = sample['clip_id']
    raw_text = sample.get('raw_text', '')

    video_path = os.path.join(video_base_dir, video_id, f'{clip_id}.mp4')
    if not os.path.exists(video_path):
        return {'status': 'failed', 'reason': f'video not found: {video_path}'}

    fps = get_video_fps(video_path)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, 'audio.wav')
        if not extract_audio_from_video(video_path, wav_path):
            return {'status': 'failed', 'reason': 'audio extraction failed'}

        audio_dur = get_audio_duration(wav_path)

        import whisperx
        try:
            audio = whisperx.load_audio(wav_path)
            result = asr_model.transcribe(audio, batch_size=8)
        except Exception as e:
            return {'status': 'failed', 'reason': f'ASR failed: {e}'}

        asr_text = ' '.join(seg.get('text', '').strip() for seg in result.get('segments', []))

        wer_score = compute_wer(raw_text, asr_text)

        try:
            result_aligned = whisperx.align(
                result['segments'], align_model, align_metadata,
                audio, DEVICE, return_char_alignments=False
            )
        except Exception as e:
            return {'status': 'failed', 'reason': f'alignment failed: {e}'}

        words = []
        for seg in result_aligned.get('segments', []):
            for w in seg.get('words', []) or []:
                words.append({
                    'word': w.get('word', '').strip(),
                    'start': w.get('start', None),
                    'end': w.get('end', None),
                    'score': w.get('score', None),
                })

        raw_words = raw_text.split()
        asr_words = [w['word'].replace(' ', '') for w in words if w['start'] is not None]
        match_map = _dp_match_words(raw_words, asr_words)

        token_records = []
        token_records.append({
            'idx': 0, 'token': '[CLS]', 'char_start': 0, 'char_end': 0,
            'audio_start_sec': 0.0, 'audio_end_sec': 0.0,
            'video_frame_start': 0, 'video_frame_end': 0,
            'asr_word': None, 'asr_confidence': None,
        })

        char_cursor = 0
        for tok_idx, raw_word in enumerate(raw_words):
            char_start = raw_text.find(raw_word, char_cursor)
            if char_start == -1:
                char_start = char_cursor
            char_end = char_start + len(raw_word)
            char_cursor = char_end + 1

            asr_idxs = match_map.get(tok_idx, [])
            if asr_idxs:
                segs = [words[i] for i in asr_idxs if words[i]['start'] is not None]
                if segs:
                    audio_start = min(s['start'] for s in segs)
                    audio_end = max(s['end'] for s in segs)
                    confs = [s['score'] for s in segs if s['score'] is not None]
                    avg_conf = float(np.mean(confs)) if confs else None
                    asr_word = ' '.join(words[i]['word'] for i in asr_idxs)
                else:
                    audio_start = audio_end = None
                    avg_conf = None
                    asr_word = None
            else:
                if audio_dur > 0 and len(raw_text) > 0:
                    audio_start = (char_start / len(raw_text)) * audio_dur
                    audio_end = (char_end / len(raw_text)) * audio_dur
                else:
                    audio_start = audio_end = 0.0
                avg_conf = None
                asr_word = None

            if audio_start is None:
                audio_start = 0.0
            if audio_end is None:
                audio_end = audio_start

            token_records.append({
                'idx': tok_idx + 1,
                'token': raw_word,
                'char_start': int(char_start),
                'char_end': int(char_end),
                'audio_start_sec': round(float(audio_start), 4),
                'audio_end_sec': round(float(audio_end), 4),
                'video_frame_start': int(audio_start * fps),
                'video_frame_end': int(audio_end * fps),
                'asr_word': asr_word,
                'asr_confidence': round(avg_conf, 4) if avg_conf is not None else None,
            })

        while len(token_records) < max_seq_len:
            last = token_records[-1]
            token_records.append({
                'idx': len(token_records),
                'token': '[PAD]',
                'char_start': last['char_end'],
                'char_end': last['char_end'],
                'audio_start_sec': float(audio_dur),
                'audio_end_sec': float(audio_dur),
                'video_frame_start': int(audio_dur * fps),
                'video_frame_end': int(audio_dur * fps),
                'asr_word': None, 'asr_confidence': None,
            })
        token_records = token_records[:max_seq_len]

        if wer_score <= 0.3:
            status = 'aligned'
        elif wer_score <= 0.6:
            status = 'partial'
        else:
            status = 'partial'

        out = {
            'video_id': video_id,
            'clip_id': clip_id,
            'sample_id': f"{video_id}$_${clip_id}",
            'asr_verification': {
                'raw_text': raw_text,
                'asr_text': asr_text,
                'wer': wer_score,
                'match_ok': wer_score <= 0.3,
            },
            'audio_duration_sec': round(float(audio_dur), 4),
            'video_fps': round(float(fps), 4),
            'alignment_method': 'whisperx-base + wav2vec2',
            'alignment_model': {
                'name': 'whisperx/wav2vec2-base-960h',
            },
            'tokens': token_records,
            'status': status,
            'created_at': datetime.now().isoformat(),
        }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser(description='whisperx 强制对齐 100 条样本')
    parser.add_argument('--input', type=str,
                        default='data/processed/features_q1/single')
    parser.add_argument('--video_dir', type=str,
                        default='data/raw_attachment1/MOSEI数据集部分原始视频-100条')
    parser.add_argument('--output', type=str,
                        default='data/processed/features_q1/alignment')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v2'])
    parser.add_argument('--limit', type=int, default=0,
                        help='只处理前 N 条（0 表示全部）')
    args = parser.parse_args()

    pkl_files = sorted(Path(args.input).glob('q1_*.pkl'))
    if args.limit > 0:
        pkl_files = pkl_files[:args.limit]
    print(f"共 {len(pkl_files)} 条样本待对齐")

    todo = []
    for p in pkl_files:
        stem = p.stem.replace('q1_', '')
        out_path = Path(args.output) / f"{stem}.json"
        if out_path.exists():
            print(f"  跳过已存在: {out_path.name}")
            continue
        todo.append((p, out_path))

    print(f"待处理: {len(todo)} 条")

    if len(todo) == 0:
        print("全部已对齐")
        return

    asr_model, align_model, align_metadata = load_whisperx_models(args.model_size)
    results = []
    for pkl_path, out_path in tqdm(todo, desc='对齐进度'):
        try:
            r = align_one_sample(
                str(pkl_path), args.video_dir,
                asr_model, align_model, align_metadata,
                str(out_path)
            )
            results.append((pkl_path.stem, r.get('status', 'failed'), r.get('asr_verification', {}).get('wer', -1)))
        except Exception as e:
            print(f"\n  X {pkl_path.stem}: {e}")
            results.append((pkl_path.stem, 'failed', -1))

    aligned = sum(1 for _, s, _ in results if s == 'aligned')
    partial = sum(1 for _, s, _ in results if s == 'partial')
    failed = sum(1 for _, s, _ in results if s == 'failed')
    print(f"\n汇总: aligned={aligned}, partial={partial}, failed={failed}")

    summary_path = Path(args.output).parent / 'alignment_summary.json'
    summary = {
        'total': len(results),
        'aligned': aligned,
        'partial': partial,
        'failed': failed,
        'items': [
            {'sample_id': s, 'status': st, 'wer': w}
            for s, st, w in results
        ],
        'created_at': datetime.now().isoformat(),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要: {summary_path}")


if __name__ == '__main__':
    main()
