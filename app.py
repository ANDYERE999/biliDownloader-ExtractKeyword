from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import converter
import ffmpeg
import os
import translator

app = Flask(__name__)
socketio = SocketIO(app)
video_name_list = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']  # 获取用户输入的网址
        download_video_bool=request.form.get('download_video')  # 获取用户是否下载视频的选择
        convert_audio_bool=request.form.get('convert_audio')  # 获取用户是否转换音频生成文本的选择
        video_title = converter.get_video_title(url)
        if video_title is None:
            return "获取视频标题失败，请检查链接是否正确"
        
        # 下载视频和音频
        socketio.start_background_task(target=download_and_convert, url=url, video_title=video_title,download_video_bool=download_video_bool,convert_audio_bool=convert_audio_bool)
        return render_template('progress.html')
    # 返回主页
    return render_template('index.html')

def download_and_convert(url, video_title,download_video_bool,convert_audio_bool):
    # 下载视频
    if download_video_bool:
        converter.download_video(url, video_title + '.mp4', socketio)
        

    if convert_audio_bool:
        # 下载音频
        converter.download_audio(url, video_title + '.m4a', socketio)

        # 转换音频格式
        input_file = video_title + '.m4a'
        output_file = video_title + '.mp3'
        ffmpeg.input(input_file).output(output_file, format="mp3").run(overwrite_output=True)
    
        # 转文字
        model_path = 'model/small'  # 本地模型文件路径
        text_output_file = video_title + ".txt"
        converter.audio_to_text(output_file, text_output_file, model_path, socketio)
    
        
    
        # 删除mp3, m4a文件
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(input_file):
            os.remove(input_file)

        # 从文件中提取关键词
        keywords=converter.extract_keywords(text_output_file, 5)
        print("提取的关键词：",keywords)
        socketio.emit('progress', {'data': '提取的关键词：' + str(keywords)})
        # 关键词写入文件
        with open(text_output_file, 'a', encoding='utf-8') as f:
            f.write("\n\n自动提取的关键词：\n")
            for keyword in keywords:
                f.write(keyword + '\n')
        
    if (download_video_bool==True) or (convert_audio_bool==True):
        os.startfile(os.getcwd())
    else:
        socketio.emit('progress',{'data':'似乎没有进行任何操作'})
    video_name_list.append(video_title)
    print("已下载的视频列表：", video_name_list,"已下载视频总数",len(video_name_list),"当前下载的视频：", video_title)

    translator.translate_from_text_file(text_output_file, video_title + "_translated.txt", src_lang="en", tgt_lang="zh-cn")

    socketio.emit('progress', {'data': '下载并转文字完成'})
    socketio.emit('task_done', {'data': '任务完成'})


if __name__ == '__main__':
    socketio.run(app,debug=True)

###后续计划：
###1.提取转文字中的关键词，调取api对文档进行总结
###2.将转文字的结果保存到数据库中，实现多用户共享
###3.建立config文件，将一些参数放入其中，方便修改
###4.建立TargetVideoList.json，实现批量下载+提取 
###5.优化网页，实现利用网页直接修改参数，例如：是否保留视频呢？是否保留音频呢？是否保留文本呢？等等 complete
###6.增加代理设置
###7.提供下载速度
###8.尝试进行硬件加速
###9.增加日志记录
###10.尝试多线程下载