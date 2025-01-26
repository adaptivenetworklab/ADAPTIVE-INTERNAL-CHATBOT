from flask_restful import Api

from chatbot_api.resources import ChatResource
from chatbot_api.resources import SessionResource

def register_routes(api: Api):
    api.add_resource(ChatResource, '/chat')
    api.add_resource(SessionResource, '/sessions')