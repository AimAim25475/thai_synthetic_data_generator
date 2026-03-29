# ตัวอย่างของ Chatbot API
# run with
#   uvicorn --host 0.0.0.0 --reload --port 3000 bot_api:app

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0,1")

import libs.Chitchat as cc

DISABLE_QA = os.getenv("DISABLE_QA", "").strip().lower() in {"1", "true", "yes"}

try:
    import libs.Classification as cf
except Exception as exc:  # noqa: BLE001
    cf = None
    print(f"Classification disabled (import failed): {exc}")

if not DISABLE_QA:
    try:
        import libs.QA as qa
    except Exception as exc:  # noqa: BLE001
        qa = None
        print(f"QA disabled (import failed): {exc}")
else:
    qa = None


app = FastAPI()

chat_history = []

@app.get("/chat")
def echo(
    line: str,
    mode: str | None = None,
    reset: bool = False,
    ret_tk: int = 3,
    red_tk: int = 1,
):
    global chat_history

    if reset:
        chat_history = []

    predicted_mode = cf.predict(line) if cf is not None else "chat_mode"
    requested_mode = (mode or "").strip().lower()
    if requested_mode in {"chat", "chat_mode"}:
        predicted_mode = "chat_mode"
    elif requested_mode in {"qa", "qa_mode"}:
        predicted_mode = "qa_mode"

    user_input = f'QUEATION: {line} </s>'
    # user_input = f'{line} </s>'
    chat_history.append( user_input)

    while len(chat_history) > 5:
        chat_history.pop(0)

    if predicted_mode == 'chat_mode' or qa is None:
        text = cc.chat(user_input=user_input, chat_history=chat_history)
    else:
        text = qa.predict(quest=line, ret_tk=ret_tk, red_tk=red_tk)

    bot_output = f"ANSWER: {text} </s>"
    # bot_output = f"{text} </s>"

    chat_history.append(bot_output)

    if text is None:
        text = "ไม่พบคำตอบ"
    return PlainTextResponse(text)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

if __name__ == '__main__':
    main()