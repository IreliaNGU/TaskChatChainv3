'''
Author: hzl
Date: 2023-08-18 15:44:53
LastEditors: hzl
LastEditTime: 2023-08-18 15:45:00
FilePath: /hzl/chat_api/server.py
Description: 
'''
import argparse
import threading
import traceback
from typing import Dict

from loguru import logger

from services.user import User

class Server:

    def __init__(self, args: argparse.Namespace):
        self.user_response = None
        self.user_id = None
        self.event = threading.Event()
        self.user_map: Dict[str, User] = dict()
        self.args = args

    def send_user_response(self, session_id, user_id, user_response):

        self.user_response = user_response
        self.user_id = user_id
        self.session_id = session_id
        if user_id not in self.user_map:
            self.user_map[user_id] = User(self.args)
        user = self.user_map.get(user_id)
        session = user.get_session(session_id)
        session.user_input_list.append(user_response)
        self.event.set()  # 设置事件，通知服务端有新的用户响应

    def receive_agent_response(self, session_id, user_id):

        try:
            user = self.user_map.get(user_id)
            session = user.get_session(session_id)
        except Exception as e:
            logger.error(e)
            raise ValueError("User not exists.")

        if session.wrong:
            # raise ValueError("Server error.") 前端应修改
            session.wrong = False
            return "抱歉，该对话发生崩溃：%s。您可以继续提问。" % str(
                session.wrong_info), session.latest_slot_dict

        return session.fetch_new_output(), session.latest_slot_dict

    def run_task_chat_chain(self):
        while True:
            self.event.wait()  # 等待事件触发，即等待有新的用户响应
            user = self.user_map.get(self.user_id)
            session = user.get_session(self.session_id)
            bot = session.bot

            try:
                agent_response, slot_dict = bot.run_text(session.latest_message)
                session.new_response_arrive(agent_response, slot_dict)
                logger.info("Agent response arrive in user_id:%s session_id:%s: %s" %
                            (str(self.user_id), str(self.session_id), agent_response))
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
                session.wrong_info = e
                session.wrong = True

            self.event.clear()