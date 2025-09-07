import random
import json

#loading responses from json file
with open(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\B_OOP\chatbot exercise\responses_chatbot.json","r") as f:
    responses = json.load(f)
    
    
#model
class Response:
    def get_response(self,user_input):
        for key in responses:
            if key in user_input:
                return random.choice(responses[key])
        
        return random.choice(responses["default"])
    

