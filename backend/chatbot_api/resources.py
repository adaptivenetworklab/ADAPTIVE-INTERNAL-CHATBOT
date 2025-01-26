from flask import request, jsonify
from flask_restful import Resource
from chatbot_api import conversational_rag_chain
from chatbot_api.sessions import storage

class ChatResource(Resource):
    def post(self):
        data = request.get_json()
        session_id = data.get('session_id')
        input_msg =  data.get('input')

        conversation_status = 'ongoing'

        def end_conversation(status):
            nonlocal conversation_status
            storage.pop(session_id)
            conversation_status = status

        if input_msg.lower() == '!selesaikan!':
            end_conversation('ended')
            return jsonify({"message": "Bye, let's have a chat again another time!", "conversation_status": conversation_status})

        response = conversational_rag_chain.invoke(
            {"input": input_msg},
            config={
                "configurable": {"session_id": session_id}
            }
        )

        response_msg = response["answer"]
        
        return jsonify({"message": response_msg, "conversation_status": conversation_status})
    
class SessionResource(Resource):
    def get(self):
        sessions = list(storage.keys())
        return jsonify({"message": sessions})