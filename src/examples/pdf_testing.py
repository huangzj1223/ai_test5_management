from langchain_pymupdf4llm import PyMuPDF4LLMLoader

file_path = "llm_course.pdf"
loader = PyMuPDF4LLMLoader(file_path, mode="single")
d = loader.load()
print(d)