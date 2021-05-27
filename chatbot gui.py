# Creating GUI with tkinter
from tkinter import *

from build_model import enc_model, str_to_tokens, tokenizer, dec_model, np, maxlen_answers

def response(msg):
	for _ in range(100):
		states_values = enc_model.predict(
			str_to_tokens(msg))
		empty_target_seq = np.zeros((1, 1))
		empty_target_seq[0, 0] = tokenizer.word_index['start']
		stop_condition = False
		decoded_translation = ''
		while not stop_condition:
			dec_outputs, h, c = dec_model.predict([empty_target_seq]
			                                      + states_values)
			sampled_word_index = np.argmax(dec_outputs[0, -1, :])
			sampled_word = None
			for word, index in tokenizer.word_index.items():
				if sampled_word_index == index:
					if word != 'end':
						decoded_translation += ' {}'.format(word)
					sampled_word = word

			if sampled_word == 'end' \
					or len(decoded_translation.split()) \
					> maxlen_answers:
				stop_condition = True

			empty_target_seq = np.zeros((1, 1))
			empty_target_seq[0, 0] = sampled_word_index
			states_values = [h, c]

		print(decoded_translation)
		return decoded_translation

def send():
	msg = EntryBox.get("1.0", 'end-1c').strip()
	EntryBox.delete("0.0", END)
	if msg != '':
		ChatLog.config(state=NORMAL)
		ChatLog.insert(END, "You: " + msg + '\n\n')
		ChatLog.config(foreground="#122240", font=("Verdana", 12))
		res = response(msg)
		ChatLog.insert(END, "Bot: " + res + '\n\n')
		ChatLog.config(state=DISABLED)
		ChatLog.yview(END)


base = Tk()
base.title("Welcome To Friendy Chatbot!")
base.geometry("560x450")
base.resizable(width=False, height=False)
# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="55", font="Arial", )
ChatLog.config(state=DISABLED)
# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
# Create Button to send message
send_btn_img= PhotoImage(file='send.png')

#SendButton = Button(base,Image=send_btn_img, font=("Helvetica", 12, 'bold'), text="Send", width="12", height=5,
                   # bd=0, bg="#8b98f0", activebackground="#3c9d9b", fg='#ffffff',
                    #command=send)
SendButton= Button(base, image=send_btn_img,command= send,
borderwidth=2)
SendButton.pack(pady=30)
# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.insert('1.0','Enter your message: ')
# EntryBox.bind("<Return>", send)
# Place all components on the screen
scrollbar.place(x=520, y=6, height=386)
ChatLog.place(x=6, y=6, height=386)
EntryBox.place(x=62, y=401, height=45, width=450)
SendButton.place(x=10, y=401, height=45)
base.mainloop()



