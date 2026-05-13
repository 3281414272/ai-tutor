# AI 思维导师系统

这是一个基于FSM + CoT + RAG的AI教育助手系统。

## 安装和运行步骤

1. 确保安装Python 3.8+（推荐3.10）

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

3. 修改API配置：
   - 打开main.py
   - 修改`OPENAI_API_KEY`为你的API密钥
   - 如果使用其他模型，修改`OPENAI_API_BASE`和`MODEL_NAME`

4. 运行程序：
   ```
   python main.py
   ```

5. 在浏览器中打开 http://127.0.0.1:7860

## 注意事项

- 需要有效的OpenAI兼容API密钥
- 首次运行会初始化向量数据库，可能需要几秒钟
- 如果遇到网络问题，请检查API配置

## 功能说明

- FSM状态机：引导学习过程
- RAG检索：基于本地知识库提供准确答案
- CoT思维链：逐步引导思考

享受学习！