import gradio as gr
from dotenv import load_dotenv
from typing import Tuple
from src.answering.answer import answer_question

load_dotenv(override=True)


def format_context(context: list) -> str:
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['type']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history: list) -> Tuple[list, str]:
    """Answer question based on contest.
    Args:
        history (list): all the messages from the chat including the latest question. 
    Return:
        history: all messages including the latest answer
        context: the formated context from the retrieved chunks. 
    """
    print("===============all messages on chatbot==============")
    print(history)
    lastest_message = history[-1]["content"]  # [{"text": ..., "type":...}]
    latest_question = lastest_message[0]["text"]
    print("==========latest message============")
    print(lastest_message)
    prior = history[:-1]
    print("===============history====================")
    print(prior)
    answer, context = answer_question(question=latest_question, history=prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message: str, history: list) -> Tuple[str, list]:
        """Collect all the messages in history including the new message from user."""
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Expert Answering Assistant", theme=theme) as ui:
        gr.Markdown("# 🏢 Expert Assistant\nAsk me anything about the company!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", 
                    height=400, 
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the company...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=400,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
