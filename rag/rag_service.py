from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.config_handler import prompts_conf
from langchain_core.runnables import RunnableLambda
from utils.chain_debug import print_prompt
from model.factory import chat_model
from utils.path_tools import get_abs_path


class RagSummarizeService:
    # 类变量缓存，所有实例共用
    _PROMPT_TEXT: str = None

    # 实体变量，每个实例独有
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = self._load_prompt_text()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _load_prompt_text(self) -> str:
        if self._PROMPT_TEXT is not None:
            # 避免重复创建对象的重复读文件加载，从缓存读取
            return self._PROMPT_TEXT

        path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except PermissionError:
            logger.error(f"无权限读取提示词文件：{path}")
            raise PermissionError(f"无权限读取提示词文件：{path}")
        except UnicodeDecodeError:
            logger.error(f"提示词文件编码错误（需UTF-8）：{path}")
            raise ValueError(f"提示词文件编码错误（需UTF-8）：{path}")
        except Exception as e:
            logger.error(f"读取提示词文件失败：{str(e)}")
            raise RuntimeError(f"读取提示词文件失败：{str(e)}")

        if not prompt_text:
            logger.error(f"提示词文件内容为空：{path}")
            raise ValueError(f"提示词文件内容为空：{path}")

        # 记录缓存
        self._PROMPT_TEXT = prompt_text
        return prompt_text


    # 对于 prompt_template: 
    #   invoke(dict) -> PromptValue 对象。
    #   内部逻辑：输入 dict，生成 PromptValue。随后模型会根据自身需求调用 
    #   PromptValue 的 to_messages() (得到 list[BaseMessage]) 或 to_string() (得到 str)，Langchain会根据模型的需求选择合适的方法,不需要手动指定。

    # 对于 model (以 ChatModel 为例): 
    #   invoke(str | list[BaseMessage] | PromptValue) -> BaseMessage 对象 (通常是 AIMessage)。
    #   注意：虽然它可以接收 str，但输出始终是一个包含元数据的 Message 对象，而非纯字符串。

    # 对于 StrOutputParser: 
    #   invoke(BaseMessage) -> str。
    #   内部逻辑：提取输入对象（如 AIMessage）的 .content 属性，将其转换为纯字符串。

    def _init_chain(self):
        # 数据流：dict -> PromptValue -> AIMessage -> str
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    # 从向量数据库检索文档，返回的是list[Document],共有k个文档
    def retrieve_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        # key: input, for user query
        # key: context, 参考资料
        input_dict = {}

        context_docs = self.retrieve_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        input_dict["input"] = query
        input_dict["context"] = context

        return self.chain.invoke(input_dict)


# for testing
if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪种扫地机器人？"))
