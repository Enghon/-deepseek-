# main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import re
import asyncio

from starlette.responses import HTMLResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化Ollama模型（显式关闭verbose模式避免额外输出）
chat_model = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://127.0.0.1:11434",
    temperature=0.7,
    verbose=False  # 关闭LangChain的调试输出
)


def clean_response(text: str) -> str:
    """
    彻底清理响应文本中的标签和异常字符
    功能：
    1. 移除<think>和</think>标签（包括拼写错误的</hink>）
    2. 处理多行内容和嵌套标签
    3. 清理多余空白字符
    """
    # 移除所有think标签变体（包括拼写错误）
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?hink>', '', text, flags=re.IGNORECASE)  # 处理可能的拼写错误

    # 移除其他可能的AI内部标记（根据实际观察添加）
    text = re.sub(r'$$.*?$$', '', text)  # 移除方括号内容
    text = re.sub(r'$$.*?$$', '', text)  # 移除圆括号内容

    # 清理多余空白（保留单个换行）
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    text = re.sub(r'\n\s+\n', '\n', text)  # 清理空行
    return text.strip()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat_stream(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    async def generate():
        messages = [HumanMessage(content=user_message)]
        full_response = ""
        async for chunk in chat_model.astream(messages):
            content = chunk.content
            # 实时清理响应内容
            cleaned_content = clean_response(content)
            if cleaned_content and cleaned_content != full_response[-len(cleaned_content):]:
                full_response += cleaned_content
                yield f"data: {cleaned_content}\n\n"
                await asyncio.sleep(0.02)  # 控制流式速度

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}  # 禁用Nginx缓冲
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8083)