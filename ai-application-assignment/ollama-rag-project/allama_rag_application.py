import streamlit as st
import os
import tempfile
from allama_rag_script import (
    extract_text_from_notebook,
    create_html_project_rag_chain,
)


# --- Streamlit UI 구성 ---
st.title("🦙 Ollama RAG - Notebook 분석")
st.markdown("Jupyter Notebook(.ipynb)을 업로드하고 내용에 대해 질문해보세요!")


# 파일 업로더 (클릭 또는 드래그 앤 드랍) - .ipynb 전용
uploaded_file = st.file_uploader(
    "Jupyter Notebook (.ipynb) 파일을 업로드하세요 (클릭 또는 드래그 앤 드랍)",
    type=["ipynb"],
)

# 메인 화면 구성
if uploaded_file is None:
    st.info("좌측의 업로더에 .ipynb 파일을 드래그 앤 드랍하거나 클릭하여 선택하세요.")
else:
    try:
        # 트리거 버튼: 사용자가 눌러야 분석 실행
        file_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 'NA')}"
        run_analysis = st.button("분석 실행")

        # 세션 상태 초기화
        if "rag_chain" not in st.session_state:
            st.session_state["rag_chain"] = None
            st.session_state["file_key"] = None

        # 사용자가 버튼을 눌렀거나, 이미 동일 파일에 대한 체인이 존재하면 사용
        if run_analysis or (
            st.session_state["rag_chain"] and st.session_state["file_key"] == file_key
        ):
            # 버튼을 눌렀고 파일이 바뀌었으면 재생성
            if run_analysis and st.session_state["file_key"] != file_key:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    notebook_path = tmp.name

                with st.spinner("노트북을 분석하고 인덱스를 생성하는 중입니다..."):
                    project_analysis = extract_text_from_notebook(notebook_path)
                    rag_chain, retriever = create_html_project_rag_chain(
                        [project_analysis]
                    )
                    st.session_state["rag_chain"] = rag_chain
                    st.session_state["file_key"] = file_key
                st.success("분석이 완료되었습니다. 이제 질문을 입력해 보세요.")

            # 사용자 질문 입력란
            question = st.text_input(
                "질문을 입력하세요:", placeholder="이 노트북의 주요 내용은 무엇인가요?"
            )

            if question and st.session_state["rag_chain"]:
                with st.spinner("답변을 생성하는 중입니다..."):
                    response = st.session_state["rag_chain"].invoke({"input": question})

                st.write("### 🤖 AI 답변:")
                # 주석(#)이 Markdown 헤더로 렌더링되지 않도록 순수 텍스트로 출력
                st.text(response.get("answer", ""))

                with st.expander("RAG Context 확인하기"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.markdown(
                            f"**문서 #{i+1}** — {doc.metadata.get('source', 'uploaded notebook')}"
                        )
                        # 코드/주석이 포함된 컨텍스트는 코드 블록으로 렌더링하여 헤더 변환 방지
                        st.code(doc.page_content, language="python")
                        st.markdown("---")
        else:
            st.info("파일을 업로드한 뒤 '분석 실행' 버튼을 눌러 분석을 시작하세요.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
