import argparse
from typing import Dict, List
from services.session import Session

class User:
    sessions: Dict[str, Session] = {}
    user_info: dict = {}
    args: argparse.Namespace

    def __init__(self, args):
        self.sessions: Dict[str, Session] = {}
        self.args = args

    def get_session(self, session_id):
        if session_id not in self.sessions:
            session = Session(self.args)
            self.sessions[session_id] = session
        session = self.sessions.get(session_id)
        return session