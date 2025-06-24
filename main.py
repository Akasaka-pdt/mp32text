import streamlit as st
import whisper
import tempfile
import io
import pandas as pd

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

# 書き起こし結果を格納するリスト
results = []

if uploaded_files:
    for file in uploaded_files:
        st.audio(file, format="audio/mp3")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        st.write(f"ファイル名: {file.name}")
        st.write(f"ファイルサイズ: {file.size / 1024:.2f} KB")

        status = st.info("文字起こし中です...")
        try:
            result = model.transcribe(tmp_path)
            status.empty()
            st.success("書き起こし完了！")

            default_text = result["text"].replace("クラシャ", "コラショ")
            edited_text = st.text_input(label=f"音声テキスト - {file.name}", value=f"{default_text}")

            # 結果をリストに追加（ファイル名とテキスト）
            results.append({"ファイル名": file.name, "書き起こしテキスト": edited_text})

        except:
            status.empty()
            st.warning(f"音声化に失敗しました。ファイルが壊れていないかご確認ください。 - {file.name}")

        st.markdown("---")

    # すべての結果をCSV化してダウンロード可能に
    if results:
        df = pd.DataFrame(results)
        csv_data = df.to_csv(index=False, encoding="utf-8-sig")  # Excel互換のBOM付き

        st.download_button(
            label="📄 書き起こし結果を CSV でダウンロード",
            data=csv_data.encode("utf-8-sig"),
            file_name="transcriptions.csv",
            mime="text/csv"
        )
