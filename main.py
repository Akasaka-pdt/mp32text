import streamlit as st
import whisper
import tempfile
import io
import zipfile

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

st.title("🎤 Whisper 音声書き起こしアプリ")

uploaded_files = st.sidebar.file_uploader(
    "音声ファイルをアップロードしてください",
    type=["mp3"],
    accept_multiple_files=True
)

# ZIP 用の in-memory buffer
zip_buffer = io.BytesIO()

if uploaded_files:
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in uploaded_files:
            st.audio(file, format="audio/mp3")

            # 一時ファイルとして保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            st.write(f"ファイル名: {file.name}")
            st.write(f"ファイルサイズ: {file.size / 1024:.2f} KB")

            # 書き起こし処理
            status = st.info("文字起こし中です...")
            result = model.transcribe(tmp_path)
            status.empty()
            st.success("書き起こし完了！")

            # 編集可能なテキストエリア
            default_text = result["text"].replace("クラシャ", "コラショ")
            edited_text = st.text_input(label=f"音声テキスト - {file.name}", value=f"{default_text}")

            # テキストを ZIP に追加
            text_filename = file.name.rsplit(".", 1)[0] + ".txt"
            zip_file.writestr(text_filename, edited_text)

            st.markdown("---")

    # ZIP ダウンロードボタン（処理後）
    zip_buffer.seek(0)
    st.download_button(
        label="📦 すべてのテキストを ZIP でダウンロード",
        data=zip_buffer,
        file_name="transcriptions.zip",
        mime="application/zip"
    )
