# -*- coding: utf-8 -*-
import streamlit as st
import whisper
import tempfile
import io
import os
import gc
import pandas as pd

# =========================
# セキュリティ強化ポイント
# - モデルのみcache_resource（ユーザーデータはキャッシュしない）
# - アップロード音声はBytesIOで扱い、必要最小限だけ一時ファイル化
# - 一時ファイルは try/finally で確実に削除
# - 例外内容は詳細を出さずに短文化（データ露出防止）
# - 拡張子/サイズ/MIMEを基本検証
# - 文字列編集はtext_area（長文＆誤入力の誤送信を減らす）
# =========================

# ▼ 追加の基本設定（.streamlit/config.toml推奨）
# [server]
# enableXsrfProtection = true
# enableCORS = false
# maxUploadSize = 50
# [browser]
# gatherUsageStats = false

ALLOWED_EXT = {".mp3"}          # 必要なら .wav などを追加
MAX_MB_EACH = 50                # 個別ファイル上限（MB）
ALLOWED_MIME = {"audio/mpeg"}   # ブラウザが付けるMIME（参考）

@st.cache_resource
def load_model():
    # Whisperローカルモデル（外部送信なし）
    return whisper.load_model("small")

model = load_model()

st.title("🎤 Whisper 音声書き起こしアプリ（堅牢版）")

uploaded_files = st.sidebar.file_uploader(
    "音声ファイルをアップロードしてください（mp3）",
    type=["mp3"],
    accept_multiple_files=True
)

# 書き起こし結果
results = []

def _safe_ext(name: str) -> str:
    # 拡張子の簡易チェック用（小文字化）
    return os.path.splitext(name or "")[1].lower()

def _bytesio_from_uploaded(file) -> io.BytesIO:
    # UploadedFile → BytesIO（二度読み防止のため一括取得）
    file.seek(0)
    data = file.read()
    return io.BytesIO(data)

def transcribe_from_bytesio(bio: io.BytesIO, suffix=".mp3") -> str:
    """
    Whisperはファイルパス入力が安定。
    一時ファイルを書いてFFmpeg→Whisper、終了後に削除。
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(bio.getbuffer())
            tmp_path = tmp.name
        result = model.transcribe(tmp_path)
        text = result.get("text", "")
        return text
    finally:
        # 一時ファイルは確実に削除
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if uploaded_files:
    for file in uploaded_files:
        # ---------- 入力検証 ----------
        # サイズ検証
        size_mb = (file.size or 0) / (1024 * 1024)
        if size_mb > MAX_MB_EACH:
            st.warning(f"❗ ファイルが大きすぎます（{size_mb:.1f}MB > {MAX_MB_EACH}MB）: {file.name}")
            continue

        # 拡張子検証
        ext = _safe_ext(file.name)
        if ext not in ALLOWED_EXT:
            st.warning(f"❗ 未対応の拡張子です: {file.name}")
            continue

        # MIME検証（参考：ブラウザ依存で厳密ではない）
        if getattr(file, "type", None) and file.type not in ALLOWED_MIME:
            # 厳しくしすぎると一部ブラウザで弾く可能性→警告に留める
            st.info(f"ℹ️ 参考情報: このファイルのMIMEタイプは {file.type} です。再生に問題がなければ続行します。")

        # ---------- 再生表示用にBytesIOを確保 ----------
        try:
            audio_bytes = _bytesio_from_uploaded(file)
        except Exception:
            st.warning(f"❗ ファイル読み込みに失敗しました。破損していないかご確認ください: {file.name}")
            continue

        # 再生（BytesIOを渡す）
        st.audio(audio_bytes, format="audio/mp3")

        # ファイルメタ表示（内容は出さない）
        st.write(f"ファイル名: {file.name}")
        st.write(f"ファイルサイズ: {size_mb:.2f} MB")

        # ---------- 文字起こし ----------
        with st.spinner("文字起こし中です…"):
            try:
                # 変換用に再度BytesIO（位置リセット）
                audio_bytes.seek(0)
                text = transcribe_from_bytesio(audio_bytes, suffix=ext)
            except Exception:
                st.warning(f"❗ 書き起こしに失敗しました。音声形式やファイル状態をご確認ください: {file.name}")
                continue
            finally:
                # メモリクリーンアップ
                del audio_bytes
                gc.collect()

        st.success("書き起こし完了！")

        # 自動置換（業務ルールに応じて調整）
        safe_text = (text or "").replace("クラシャ", "コラショ")

        # 編集はtext_areaに（長文対応＆誤送信抑止）
        edited_text = st.text_area(
            label=f"音声テキスト - {file.name}",
            value=safe_text,
            height=180
        )

        results.append({"ファイル名": file.name, "書き起こしテキスト": edited_text})
        st.markdown("---")

    # ---------- すべての結果をCSV化 ----------
    if results:
        try:
            df = pd.DataFrame(results)
            csv_data = df.to_csv(index=False, encoding="utf-8-sig")  # Excel互換
            st.download_button(
                label="📄 書き起こし結果を CSV でダウンロード",
                data=csv_data.encode("utf-8-sig"),
                file_name="transcriptions.csv",
                mime="text/csv"
            )
        except Exception:
            st.warning("❗ 結果のエクスポートに失敗しました。もう一度お試しください。")
        finally:
            # 大きなオブジェクトを解放
            del results
            gc.collect()
else:
    st.info("サイドバーから音声ファイル（mp3）をアップロードしてください。")
