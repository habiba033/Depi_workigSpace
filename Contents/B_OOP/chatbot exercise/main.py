from chatpot_model import Response

r1 = Response()

if __name__ == "__main__":
    print("Chatbot : How can i assist you today ?")
    while True:
        user_input = input("User : ").lower()
        response = r1.get_response(user_input)
        print("Chatbot : ", response)
        
        if user_input == "goodbye":
            break
        