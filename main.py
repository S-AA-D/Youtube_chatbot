from chatbot import YoutubeChatBot

chatbot = YoutubeChatBot(youtube_url="ENTER THE VIDEO URL HERE" ) # If this was your first interact with the model it will transcript the video and this process may take several minutes depending on the video long

while True:
  question = input("Enter your question : ")
  print("Chatbot: "+chatbot.ask(question))
