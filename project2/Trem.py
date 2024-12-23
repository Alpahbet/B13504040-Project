import pyttsx3
import uuid
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from googletrans import Translator
import os
import soundfile as sf

app = Flask(__name__)
CORS(app)

# 初始化 text-to-speech 引擎
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# 设置英语语音
def set_english_voice():
    english_voice = None
    for voice in voices:
        if 'english' in voice.languages:
            english_voice = voice
            break
    if english_voice:
        engine.setProperty('voice', english_voice.id)
    else:
        engine.setProperty('voice', voices[0].id)

# 将文本转换为语音并保存到文件
def speak_to_file(text, filename):
    try:
        set_english_voice()
        engine.save_to_file(text, filename)
        engine.runAndWait()
        return filename
    except Exception as e:
        return str(e)

# 翻译函数
def translate_text(input_text, src_lang='zh-TW', dest_lang='en'):
    try:
        translator = Translator()
        translation = translator.translate(input_text, src=src_lang, dest=dest_lang)
        return translation.text
    except Exception as e:
        return str(e)

# 特征提取函数，使用 librosa 或 soundfile 加载音频文件
def extract_features(audio_file):
    try:
        # 尝试使用 soundfile 读取音频文件
        try:
            y, sr = sf.read(audio_file)  # 使用 soundfile 读取
        except Exception as e:
            print(f"Error reading audio file with soundfile: {e}")
            # 如果 soundfile 失败，回退使用 librosa
            try:
                y, sr = librosa.load(audio_file, sr=None)  # 使用 librosa 加载
            except Exception as e:
                print(f"Error reading audio file with librosa: {e}")
                return None  # 如果两者都失败，则返回 None
        
        # 使用 librosa 提取 MFCC 特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# 比较两个音频文件的相似度
def compare_audio(file1, file2):
    try:
        features1 = extract_features(file1)
        features2 = extract_features(file2)
        
        if features1 is None or features2 is None:
            print("Error: One or both feature extraction failed.")
            return None
        
        # 计算两个音频文件的相似度，使用余弦相似度
        similarity = 1 - cosine(features1, features2)
        return similarity
    except Exception as e:
        print(f"Error during audio comparison: {e}")
        return None

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# API: 生成语音并保存
@app.route('/api/speak', methods=['POST'])
def api_speak():
    try:
        data = request.json
        text = data.get('text', '')
        src_lang = data.get('src_lang', 'zh-TW')
        dest_lang = data.get('dest_lang', 'en')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # 如果需要翻译文本
        translated_text = translate_text(text, src_lang=src_lang, dest_lang=dest_lang)

        if "Exception" in translated_text:
            return jsonify({'error': 'Translation failed', 'details': translated_text}), 500

        # 生成唯一的文件名
        filename = f"output_{uuid.uuid4().hex}.wav"
        audio_file = speak_to_file(translated_text, filename)

        if "Exception" in audio_file:
            return jsonify({'error': 'Text-to-speech conversion failed', 'details': audio_file}), 500

        return jsonify({'status': 'success', 'filename': filename, 'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': 'Server error during text-to-speech conversion', 'details': str(e)}), 500

# API: 比较用户上传的音频与生成的音频
@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        data = request.files
        user_audio = data.get('user_audio')
        generated_audio = data.get('generated_audio')

        if not user_audio or not generated_audio:
            return jsonify({'error': 'Both audio files are required'}), 400

        # 保存上传的音频文件
        user_audio_path = f"user_{uuid.uuid4().hex}.wav"
        generated_audio_path = f"generated_{uuid.uuid4().hex}.wav"

        user_audio.save(user_audio_path)
        generated_audio.save(generated_audio_path)

        # 输出日志，查看文件保存路径
        print(f"User audio saved to: {user_audio_path}")
        print(f"Generated audio saved to: {generated_audio_path}")

        # 比较音频文件
        similarity = compare_audio(user_audio_path, generated_audio_path)

        # 输出音频比较结果
        if similarity is None:
            print("Audio comparison failed.")
            return jsonify({'error': 'Comparison failed'}), 500

        print(f"Audio similarity: {similarity * 100}%")
        return jsonify({'status': 'success', 'similarity': round(similarity * 100, 2)})

    except Exception as e:
        print(f"Error in audio comparison: {e}")
        return jsonify({'error': 'Server error during audio comparison'}), 500

# API: 下载文件
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': 'File download failed', 'details': str(e)}), 500

# 运行应用
if __name__ == '__main__':
    app.run(debug=True)
