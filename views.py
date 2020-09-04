import os
from flask import render_template, Flask, request, redirect, url_for
try:
    import simplejson as json
except:
    import json
import tensorflow as tf
from QA_init import GiveFlaskWebData


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dir_path = os.getcwd()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session=tf.Session(config=config)
# KTF.set_session(session)

app = Flask(__name__)
# graph = tf.get_default_graph()


@app.route('/')
@app.route('/success/')
@app.route('/ownthinkQA/')
def strart():
    return render_template('index.html')


# 系统使用方式提醒
@app.route('/ownthinkQA/<question>')
def ownthinkQA(question):
    ans, data, link = search(question)
    data_01 = json.dumps(data)
    link_01 = json.dumps(link)
    return render_template('true.html', ans=ans, data_list=data_01, link_list=link_01)


@app.route('/search', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        question = request.form['question']
        print((question))
        return redirect(url_for('ownthinkQA', question=question))
    else:
        question = request.args.get('question')
        return redirect(url_for('ownthinkQA', question=question))


def search(question):
    getData = GiveFlaskWebData()
    answer, data, link = getData.getWebTypeData(question)

    # print(data)
    return answer, data, link


if __name__ == "__main__":
    app.debug = True
    app.run()
