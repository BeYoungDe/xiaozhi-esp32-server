from config.logger import setup_logging
from openai import OpenAI
import json
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        logger.bind(tag=TAG).debug("正在初始化LLMProvider")
        self.model_name = config.get("model_name")
        self.base_url = config.get("base_url", "http://172.18.120.18:11434")
        logger.bind(tag=TAG).debug(f"基础URL: {self.base_url}")
        # 初始化OpenAI客户端，使用Ollama的基础URL
        # 如果没有v1，增加v1
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
            logger.bind(tag=TAG).debug("为URL添加了'/v1'")
        logger.bind(tag=TAG).debug(f"更新后的基础URL: {self.base_url}")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama"  # Ollama不需要API密钥，但OpenAI客户端需要一个
        )
        logger.bind(tag=TAG).debug("OpenAI客户端初始化完成")

    def response(self, session_id, dialogue):
        logger.bind(tag=TAG).info(f"正在为会话{session_id}生成响应")
        try:
            logger.bind(tag=TAG).debug(f"使用模型{self.model_name}创建聊天补全")
            responses = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True
            )
            logger.bind(tag=TAG).debug("聊天补全创建成功")
            is_active = True
            for chunk in responses:
                logger.bind(tag=TAG).debug(f"正在处理数据块: {chunk}")
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, 'choices', None) else None
                    content = delta.content if hasattr(delta, 'content') else ''
                    logger.bind(tag=TAG).debug(f"Delta内容: {content}")
                    if content:
                        if '<think>' in content:
                            is_active = False
                            content = content.split('<think>')[0]
                            logger.bind(tag=TAG).debug("检测到'<think>'，停用响应")
                        if '</think>' in content:
                            is_active = True
                            content = content.split('</think>')[-1]
                            logger.bind(tag=TAG).debug("检测到'</think>'，重新启用响应")
                        if is_active:
                            logger.bind(tag=TAG).debug("正在输出活跃内容")
                            yield content
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理数据块时出错: {e}")
                    logger.bind(tag=TAG).error(f"导致错误的数据块: {chunk}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Ollama响应生成出错: {e}")
            logger.bind(tag=TAG).error(f"导致错误的对话内容: {dialogue}")
            yield "【Ollama服务响应异常】"

    def response_with_functions(self, session_id, dialogue, functions=None):
        logger.bind(tag=TAG).info(f"正在为会话{session_id}生成带函数调用的响应")
        try:
            logger.bind(tag=TAG).debug(f"使用模型{self.model_name}创建聊天补全")
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True,
                tools=functions,
            )
            logger.bind(tag=TAG).debug("聊天补全创建成功")
            for chunk in stream:
                logger.bind(tag=TAG).debug(f"正在处理数据块: {chunk}")
                try:
                    content = chunk.choices[0].delta.content
                    tool_calls = chunk.choices[0].delta.tool_calls
                    logger.bind(tag=TAG).debug(f"内容: {content}, 工具调用: {tool_calls}")
                    yield content, tool_calls
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理数据块时出错: {e}")
                    logger.bind(tag=TAG).error(f"导致错误的数据块: {chunk}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Ollama函数调用出错: {e}")
            logger.bind(tag=TAG).error(f"导致错误的对话内容: {dialogue}")
            yield f"【Ollama服务响应异常: {str(e)}】", None