import warnings
import streamlit as st
import glob
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

# HTML 파일 분석을 위한 추가 라이브러리
from langchain_community.document_loaders import UnstructuredHTMLLoader
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")


def extract(file_path):
    try:
        with open(file_path) as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""
    code_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code_cells.append("".join(cell["source"]))
        elif cell["cell_type"] == "markdown":
            code_cells.append("".join(cell["source"]))

    all_text = "\n\n".join(code_cells)

    return all_text


def extract_text_from_notebook(file_path):
    lines = extract(file_path).split("\n")

    imports = []
    functions = []
    classes = []
    code_blocks = []

    in_code_block = False
    current_code_block = []

    for line in lines:
        line = line.strip()

        # Python 코드 패턴 감지
        if (
            line.startswith("import")
            or line.startswith("from")
            or line.startswith("def")
            or line.startswith("class")
            or line.startswith("if")
            or line.startswith("for")
            or line.startswith("while")
            or "=" in line
        ):

            if line.startswith("import") or line.startswith("from"):
                imports.append(line)
            elif line.startswith("def "):
                functions.append(line)
            elif line.startswith("class "):
                classes.append(line)

            current_code_block.append(line)
            in_code_block = True
        elif in_code_block and line == "":
            if current_code_block:
                code_blocks.append("\n".join(current_code_block))
            current_code_block = []
            in_code_block = False
        elif in_code_block:
            current_code_block.append(line)

    # 마지막 코드 블록 추가
    if current_code_block:
        code_blocks.append("\n".join(current_code_block))

    # 중복 제거 및 정리
    imports = list(set([imp for imp in imports if imp]))
    functions = list(set([func for func in functions if func]))
    classes = list(set([cls for cls in classes if cls]))
    code_blocks = [block for block in code_blocks if block and len(block.strip()) > 5]

    project_type = "Unknown"
    if any("tensorflow" in imp.lower() or "keras" in imp.lower() for imp in imports):
        project_type = "Deep Learning (TensorFlow/Keras)"
    elif any("torch" in imp.lower() or "pytorch" in imp.lower() for imp in imports):
        project_type = "Deep Learning (PyTorch)"
    elif any("sklearn" in imp.lower() or "scikit" in imp.lower() for imp in imports):
        project_type = "Machine Learning (Scikit-learn)"
    elif any("pandas" in imp.lower() or "numpy" in imp.lower() for imp in imports):
        project_type = "Data Analysis"
    elif any(
        "streamlit" in imp.lower() or "flask" in imp.lower() or "fastapi" in imp.lower()
        for imp in imports
    ):
        project_type = "Web Application"

    analysis_result = {
        "project_title": os.path.basename(file_path).split(".")[0],
        "file_path": file_path,
        "project_type": project_type,
        "total_imports": len(imports),
        "total_functions": len(functions),
        "total_classes": len(classes),
        "total_code_blocks": len(code_blocks),
        "imports": imports[:15],  # 처음 15개만
        "functions": functions[:10],  # 처음 10개만
        "classes": classes[:10],  # 처음 10개만
        "code": code_blocks if code_blocks else [],
    }

    return analysis_result


def create_html_documents_for_rag(all_projects):
    """
    분석된 HTML 프로젝트들을 RAG 시스템용 문서로 변환
    """
    documents = []

    for project in all_projects:
        # 프로젝트 개요 문서 생성
        doc_content = f"""
프로젝트명: {project['project_title']}
파일 경로: {project['file_path']}
프로젝트 유형: {project['project_type']}

=== 통계 정보 ===
- 총 임포트 수: {project['total_imports']}
- 총 함수 수: {project['total_functions']}
- 총 클래스 수: {project['total_classes']}
- 총 코드 블록 수: {project['total_code_blocks']}

=== 주요 임포트 ===
{chr(10).join(project['imports'])}

=== 함수 정의 ===
{chr(10).join(project['functions'])}

=== 클래스 정의 ===  
{chr(10).join(project['classes'])}

=== 코드 예시 ===
{chr(10).join(project['code'])}
"""

        doc = Document(
            page_content=doc_content,
            metadata={
                "source": project["file_path"],
                "project_name": project["project_title"],
                "project_type": project["project_type"],
            },
        )
        documents.append(doc)

    return documents


# HTML 프로젝트용 RAG 체인 생성
def create_html_project_rag_chain(all_projects):
    """
    HTML 프로젝트 분석 결과를 기반으로 RAG 체인 생성
    """
    # 데이터 검증
    if not all_projects:
        raise ValueError("분석할 프로젝트가 없습니다.")

    print(f"총 {len(all_projects)}개 프로젝트 처리 중...")

    # 문서 생성
    documents = create_html_documents_for_rag(all_projects)
    print(f"{len(documents)}개 문서 생성됨")

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=3000, chunk_overlap=500
    )
    split_docs = splitter.split_documents(documents)
    print(f"{len(split_docs)}개 청크로 분할됨")

    # 임베딩 및 벡터 스토어 생성
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    # retriever 설정을 모든 문서를 반환하도록 수정
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": len(split_docs)}  # 모든 문서 반환
    )
    print(f"벡터 검색기 생성됨 (총 {len(split_docs)}개 문서 검색)")

    # LLM 설정
    llm = ChatOllama(model="llama3.1:8b")

    # 프롬프트 설정 (한국어 지원)
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 데이터 사이언스 및 머신러닝 프로젝트 전문가입니다. 제공된 컨텍스트를 바탕으로 프로젝트에 대한 질문에 답변해주세요.
        한국어로 답변하고, 가능한 한 구체적으로 설명하세요.
        <context>
        {context}
        </context>

        질문: {input}

        답변:
      """
    )

    # RAG 체인 생성
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain, retriever


def main():
    # RAG 체인 생성 및 테스트

    DATA_PATH = "./data"
    notebook_name = "AT"

    all_projects = [extract_text_from_notebook(f"{DATA_PATH}/{notebook_name}.ipynb")]

    print("🤖 HTML 프로젝트 분석 AI 에이전트 생성 중...")
    html_rag_chain, html_retriever = create_html_project_rag_chain(all_projects)
    print("✅ RAG 체인 생성 완료!")

    # 테스트 질문들
    test_questions = ["해당 프로젝트의 구조는 어떻게 되어있나요?"]

    print("\n질문 실행:")
    for i, question in enumerate(test_questions, 1):
        print(f"\n=== 질문 {i}: {question} ===")

        retrieved_docs = html_retriever.get_relevant_documents(question)
        print(f"검색된 문서 수: {len(retrieved_docs)}")

        # 모든 프로젝트 이름 확인
        project_names = [
            doc.metadata.get("project_name", "Unknown") for doc in retrieved_docs
        ]
        print(f"검색된 프로젝트들: {set(project_names)}")

        docount = 0
        for j, doc in enumerate(retrieved_docs, 1):  # 처음 3개만 출력
            print(f"문서 {j}의 프로젝트 이름: {doc.metadata['project_name']}")
            docount += 1

        print(f"{docount}개 문서에서 답변 생성 중...")

        response = html_rag_chain.invoke({"input": question})
        print("💬 답변:", response["answer"])
        print("-" * 50)


if __name__ == "__main__":
    main()
