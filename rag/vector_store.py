from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embed_model
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
import os
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, csv_loader, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tools import get_abs_path


class VectorStoreService:
    _loaded = False
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    # 加载文档到向量数据库
    def load_document(self):
        if self._loaded:
            logger.info("[加载知识库] 已加载，跳过")
            return

        # 检查文件是否已经存在于知识库中
        def check_md5_hex(md5_for_check):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

            return False

        # 保存文件MD5哈希值到文件
        def save_md5_hex(md5_for_save):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save+"\n")

        # 从文件路径加载文档
        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith("csv"):
                return csv_loader(read_path)
            else:
                return []

        allowed_files_path = listdir_with_allowed_type(
            # 需要是绝对路径，否则会报错
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if not md5_hex:  # 处理MD5计算失败的情况
                logger.warning(f"[加载知识库] {path} MD5计算失败，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {path} 内容已经存在于知识库，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库] {path} 无有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库] {path} 分片后无内容，跳过")
                    continue
                
                # 加载文档分片到向量数据库，但是md5值是整个文件的md5值，而不是每个分片的md5值
                self.vector_store.add_documents(split_document)

                # 保存文件MD5哈希值到文件
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库] {path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细报错堆栈，False仅记录报错str
                logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
                continue
        
        self._loaded = True



# for testing
if __name__ == '__main__':
    store = VectorStoreService()
    store.load_document()

    retriever = store.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
