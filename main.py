import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # 返回固定维度的随机向量
        import numpy as np
        return [np.random.rand(384).tolist() for _ in texts]
    
    def embed_query(self, text):
        import numpy as np
        return np.random.rand(384).tolist()
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# ==========================================
# 0. API 配置 (请替换为你的 Qwen 或其他大模型 API Key)
# 阿里云 DashScope Qwen 示例：
# ==========================================
os.environ["OPENAI_API_KEY"] = "sk-eeogknifmkcjsfyoctrkhrukyhtiueyohsglklfklxsgmrfw"
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"
# 如果使用其他模型，按需修改 BASE URL 和 API KEY

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"  # 或 "qwen2-7b-instruct"


# ==========================================
# 模块四：RAG 知识校准初始化
# ==========================================
def init_rag_db():
    print("正在初始化本地向量数据库...")
    # 模拟读取本地知识库
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = "C++单链表反转需要维护三个指针：prev, curr, next_temp。边界条件需考虑 head == nullptr。"

    # 文本分块与向量化
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    # 使用模拟 embeddings 避免依赖问题
    embeddings = MockEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore


vectorstore = init_rag_db()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)


# ==========================================
# 模块一 & 二：FSM 状态机与约束性提示工程
# ==========================================
class TutorFSM:
    def __init__(self):
        # 状态定义：0-初始引入, 1-思路引导, 2-代码检查, 3-边界测试
        self.state_map = {
            0: "概念引入",
            1: "思路引导",
            2: "代码检查",
            3: "边界测试"
        }

    def get_prompt_for_state(self, state, rag_context, user_input):
        """
        核心模块三：多跳思维链 (CoT) 的控制中枢
        根据不同状态，给 LLM 不同的 System Constraint
        """
        base_rule = "你是一个严厉但富有启发性的计算机专业AI导师。绝对禁止直接给出完整的代码或最终答案！"

        if state == 0:
            sys_msg = f"{base_rule} 识别用户的编程问题。根据以下知识：\n{rag_context}\n引导用户思考解决这个问题需要用到什么数据结构或核心概念。用反问句结束。"
        elif state == 1:
            sys_msg = f"{base_rule} 针对用户提出的思路。根据以下知识：\n{rag_context}\n如果思路正确，引导他们思考具体的指针操作（如需要几个指针）；如果错误，给出提示并反问。不要写代码。"
        elif state == 2:
            sys_msg = f"{base_rule} 检查用户的代码片段或伪代码。根据以下知识：\n{rag_context}\n指出逻辑漏洞，但不要直接修复，要求用户自己修改。"
        else:
            sys_msg = f"{base_rule} 引导用户思考极端情况和边界条件。根据以下知识：\n{rag_context}\n提问如果输入为空（如 head == nullptr）会发生什么。"

        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_input}]


fsm = TutorFSM()


# ==========================================
# 核心对话处理逻辑
# ==========================================
def chat_logic(user_message, history, current_state):
    # 1. 触发 RAG 检索知识
    docs = retriever.invoke(user_message)
    rag_context = "\n".join([d.page_content for d in docs]) if docs else "无相关背景知识。"

    # 2. 根据 FSM 当前状态构建 Prompt
    prompt = fsm.get_prompt_for_state(current_state, rag_context, user_message)

    # 3. 调用 LLM
    response = llm.invoke(prompt).content

    # 4. FSM 状态流转逻辑 (简单模拟，实际可通过 LLM 意图识别来推动状态)
    next_state = current_state + 1 if current_state < 3 else 0
    state_label = f"当前状态: {fsm.state_map[next_state]}"

    # 拼接历史记录，Gradio 需要 messages 格式
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})

    return history, next_state, state_label, ""


# ==========================================
# 前端展示层：Gradio Web UI
# ==========================================
with gr.Blocks(title="AI 思维导师系统") as demo:
    gr.Markdown("## 🎓 智慧教育 AI 导师演示系统 (基于 FSM + CoT + RAG)")

    # 隐式状态变量
    state_var = gr.State(value=0)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话可视化界面", height=500)
            with gr.Row():
                msg_input = gr.Textbox(label="学生输入", placeholder="例如：我想写一个C++单链表反转，给我代码。", lines=2)
                submit_btn = gr.Button("发送", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 控制台中枢")
            state_display = gr.Textbox(label="FSM 状态机追踪", value="当前状态: 概念引入", interactive=False)
            clear_btn = gr.Button("重置会话状态")

    # 事件绑定
    submit_btn.click(
        fn=chat_logic,
        inputs=[msg_input, chatbot, state_var],
        outputs=[chatbot, state_var, state_display, msg_input]
    )
    # 按回车发送
    msg_input.submit(
        fn=chat_logic,
        inputs=[msg_input, chatbot, state_var],
        outputs=[chatbot, state_var, state_display, msg_input]
    )


    def reset_state():
        return [], 0, "当前状态: 概念引入", ""


    clear_btn.click(fn=reset_state, inputs=[], outputs=[chatbot, state_var, state_display, msg_input])

if __name__ == "__main__":
    print("系统启动中... 请在浏览器中打开 http://127.0.0.1:7860")
    demo.launch(theme=gr.themes.Soft())
