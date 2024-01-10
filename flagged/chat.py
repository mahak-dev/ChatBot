import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Chatbot:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        self.messages = [
            {"role": "system", "content": "You are a conversational AI."},
        ]

    def generate_reply(self, input_text):
        if input_text:
            self.messages.append({"role": "user", "content": input_text})

            input_ids = self.tokenizer.encode("\n".join(msg["content"] for msg in self.messages), return_tensors="pt")
            output = self.model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

            reply = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.messages.append({"role": "assistant", "content": reply})
            return reply

# Create an instance of the Chatbot class
chatbot_instance = Chatbot()

inputs = gr.Textbox(lines=7, label="Chat with AI")
outputs = gr.Textbox(label="Reply")

# Create the Gradio interface
gr.Interface(fn=chatbot_instance.generate_reply, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask me anything, and I'll do my best to respond!",
             theme="compact").launch(share=True)
