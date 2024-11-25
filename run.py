from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
from loguru import logger

from services import server,args

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"
CORS(app)

lock = threading.Lock()

counter=0
MAX_CONCURRENT_REQUESTS = 20


@app.route('/send_user_message', methods=['POST'])
def send_user_message():
    global counter

    # 请求过载，返回提示信息
    if counter >= MAX_CONCURRENT_REQUESTS:
        return jsonify({'message': '请稍等再试'}), 504

    # 获取线程锁
    with lock:
        counter += 1

    try:
        #获取信息
        session_id = request.json['session_id']
        user_id = request.json['user_id']
        user_response = request.json['user_response']

        #如果客户回复过长
        if len(user_response) > 100:
            return jsonify({'message': "Request Entity Too Large"}), 413

        #正常回复
        response = {'message': "OK"}

        logger.info("successfully get message from user_id:%s,session_id:%s: %s" %
                    (user_id, session_id, user_response))

        server.send_user_response(session_id, user_id, user_response)

        return jsonify(response), 200

    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1


@app.route('/receive_agent_message', methods=['POST'])
def receive_agent_message():
    global counter

    # 请求过载，返回提示信息
    if counter >= MAX_CONCURRENT_REQUESTS:
        return jsonify({'message': '请稍等再试'}), 504

    # 获取线程锁
    with lock:
        counter += 1

    try:
        #获取信息
        session_id = request.json['session_id']
        user_id = request.json['user_id']

        try:
            agent_response, latest_slot_dict = server.receive_agent_response(
                session_id, user_id)
        except Exception as e:
            return jsonify({'message': str(e), "agent_response": "", "new_slots": {}}), 500

        #如果模型仍在生成回复
        if agent_response is None:
            return jsonify({
                'message': "Keep waiting",
                "agent_response": "",
                "new_slots": {}
            }), 200

        logger.info("send agent response and slot dict to user_id:%s session_id:%s" %
                    (user_id, session_id))

        return jsonify({
            'message': "OK",
            "agent_response": agent_response,
            "new_slots": latest_slot_dict
        }), 200

    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1


if __name__ == "__main__":
    task_chat_chain_thread = threading.Thread(target=server.run_task_chat_chain)
    task_chat_chain_thread.start()

    logger.info("服务器已启动")
    app.run(host='0.0.0.0', port=args.flask_port)