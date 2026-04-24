import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

async def process_pdf(file_path: str):
    # 设置 API 配置
    llm_api_key = "sk-e8dd8fe950d849cf9552eaeeef986359"
    llm_base_url = "https://api.deepseek.com/v1"
    vllm_api_key = "12f343c3-4ae7-4828-b071-9eeebddf5c34"
    vllm_base_url = "https://ark.cn-beijing.volces.com/api/v3"

    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="docling",  # 选择解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 定义 LLM 模型函数
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "deepseek-chat",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=llm_api_key,
            base_url=llm_base_url,
            **kwargs,
        )

    # 定义视觉模型函数用于图像处理
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # 如果提供了messages格式（用于多模态VLM增强查询），直接使用
        if messages:
            return openai_complete_if_cache(
                "doubao-seed-1-6-vision-250815",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=vllm_api_key,
                base_url=vllm_base_url,
                **kwargs,
            )
        # 传统单图片格式
        elif image_data:
            return openai_complete_if_cache(
                "doubao-seed-1-6-vision-250815",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=vllm_api_key,
                base_url=vllm_base_url,
                **kwargs,
            )
        # 纯文本格式
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定义嵌入函数
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts,
            embed_model="qwen3-embedding:0.6b",
            api_key="fake-api-key",
            timeout=600,
            host="http://104.233.187.228:11434",
        ),
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # 处理文档
    await rag.process_document_complete(
        file_path=file_path,
        output_dir="./output",
        parse_method="auto"
    )


if __name__ == "__main__":
    asyncio.run(process_pdf("接口文档.pdf"))