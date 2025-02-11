import yt_dlp
import os
import datetime
import ffmpeg
import ctypes
from faster_whisper import WhisperModel
from tqdm import tqdm
import numpy as np

# 获取视频标题
def get_video_title(url):
    ydl_opts = {
        'format': 'bestaudio/best',  # 最高音质
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        title = info_dict.get('title', None)
        return title

# 下载视频
def download_video(url, video_title=None, socketio=None):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # 指定可用的格式
        'outtmpl': video_title  # 保存文件名
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        if socketio:
            socketio.emit('progress', {'data': '视频下载完成'})

# 下载音频
def download_audio(url, video_title=None, socketio=None):
    ydl_opts = {
        'format': 'bestaudio/best',  # 最高音质
        'outtmpl': video_title  # 保存文件名
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        if socketio:
            socketio.emit('progress', {'data': '音频下载完成'})

# 转文字,并显示进度条
def audio_to_text(input_file, output_file, model_path, socketio=None):
    model = WhisperModel(model_path, device="cpu", local_files_only=True)  # 指定设备为 CPU 并仅使用本地文件
    segments, info = model.transcribe(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(segment.text + '\n')
            if socketio:
                socketio.emit('progress', {'data': f'转文字进度: {segment.start:.2f}秒 - {segment.end:.2f}秒'})

def audio_to_text_accelerate_cuda(input_file, output_file, model_path):
    model = WhisperModel(model_path, device="cuda", local_files_only=True)  # 指定设备为 GPU 并仅使用本地文件
    segments, info = model.transcribe(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in tqdm(segments, desc="Transcribing"):
            f.write(segment.text + '\n')

### 从提取出来的文字中提取关键词
from sklearn.feature_extraction.text import TfidfVectorizer
#from transformers import pipeline
from rake_nltk import Rake
from textrank4zh import TextRank4Keyword
def extract_keywords_TF_IDF(text_file, top_k=10):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.array(tfidf_matrix.todense()).flatten()
    top_indices = tfidf_scores.argsort()[-top_k:][::-1]
    keywords = [feature_names[i] for i in top_indices]
    return keywords
# def extract_keywords_bert(text_file, top_k=10):
#     summarizer = pipeline("summarization")
#     with open(text_file, 'r',encoding='utf-8') as f:
#         text = f.read()
#     keywords = summarizer(text, max_length=50, min_length=10, do_sample=False)
#     return keywords
def extract_keywords_rake(text_file, top_k=10):
    rake = Rake()
    with open(text_file, 'r',encoding='utf-8') as f:
        text = f.read()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases_with_scores()
    return ranked_phrases[:top_k]
def extract_keywords_textrank(text_file, top_k=10):
    with open(text_file, 'r',encoding='utf-8') as f:
        text = f.read()
    tr4w=TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)
    keywords = tr4w.get_keywords(topK=top_k, word_min_len=2)
    return [(item.word, item.weight) for item in keywords]
def extract_keywords(text_file_dir,top_k=10,method="auto"):
    '''
    提取关键词
    :param text_file_dir: 文本文件路径
    :param top_k: 提取关键词的数量
    :param method: 提取关键词的方法
    :共有四种方法：TF-IDF、BERT、RAKE、TextRank
    :对于auto方法，会自动选择最佳的方法，具体来说，会读取文字总数，如果文字总数小于1000，则使用TF-IDF，如果文字总数大于1000，则使用TextRank
    :return: 关键词列表
    '''
    if method=="auto":
        with open(text_file_dir, 'r',encoding='utf-8') as f:
            text = f.read()
        if len(text)<1000:
            method="TF-IDF"
        else:
            method="TextRank"
    if method=="TF-IDF":
        return extract_keywords_TF_IDF(text_file_dir,top_k)
    elif method=="BERT":
        #return extract_keywords_bert(text_file_dir,top_k)
        pass
    elif method=="RAKE":
        return extract_keywords_rake(text_file_dir,top_k)
    elif method=="TextRank":
        return extract_keywords_textrank(text_file_dir,top_k)
    else:
        raise ValueError("method参数错误")

if __name__ == '__main__':
    # 初始化socketio
    socketio = None
    # 获取视频标题
    url = input("请输入视频链接:")
    video_title = get_video_title(url)
    if video_title is None:
        print('获取视频标题失败，请检查链接是否正确')
        exit(1)
    
    # 判断音频文件是否已经存在
    if os.path.exists(video_title + '.m4a'):
        print('音频文件已存在，无需下载')
    else:
        print('开始下载音频...')
        # 下载音频
        download_audio(url, video_title + '.m4a', socketio)

    # 转换音频格式
    input_file = video_title + '.m4a'
    output_file = video_title + '.mp3'
    # 判断音频文件是否已经存在
    if os.path.exists(video_title + '.mp3'):
        print('音频文件已存在，无需转换')
    else:
        print('开始转换音频格式...')
        ffmpeg.input(input_file).output(output_file, format="mp3").run()
    
    # 转文字
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    text_output_file = f"{video_title}.txt"
    model_path = 'model/small'  # 本地模型文件路径
    # 判断文本文件是否已经存在
    if os.path.exists(text_output_file):
        print('文本文件已存在，无需转换')
    else:
        print('开始转换文字...')
        audio_to_text(output_file, text_output_file, model_path, socketio)
    
    print(f'下载并转文字完成，结果保存在 {text_output_file}')
    os.open(text_output_file, os.O_RDWR)  # 打开文件
    os.remove(video_title + '.m4a')
