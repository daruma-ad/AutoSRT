"""
app.py
======
AutoSRT - 動画編集前処理自動化ツール

Streamlit ベースの Web アプリケーション。
フェーズ1: インテリジェント無音カット機能（XML × Audio）
フェーズ2: コンテキスト考慮型 SRT 校正機能（SRT × LLM）
"""

import os
import pathlib
import streamlit as st
import numpy as np

from audio_processor import (
    process_silence_cut,
    load_audio_as_wav,
    detect_silence,
    CutResult,
)
from srt_processor import (
    process_srt_correction,
    entries_to_srt,
    CorrectionResult,
)


# ===========================================================================
# ヘルパー関数（先頭で定義 — Streamlit はトップダウンで実行されるため）
# ===========================================================================
def _format_time(seconds: float) -> str:
    """秒を MM:SS 形式にフォーマット"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _escape_html(text: str) -> str:
    """HTML エスケープ"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def _save_to_desktop(content: bytes, filename: str):
    """ファイルをデスクトップに直接書き込む"""
    desktop = pathlib.Path.home() / "Desktop"
    save_path = desktop / filename
    try:
        with open(save_path, "wb") as f:
            f.write(content)
        st.success(f"✅ デスクトップに保存しました: `{filename}`")
    except Exception as e:
        st.error(f"❌ デスクトップへの保存に失敗しました: {e}")


def _get_api_key() -> str:
    """Gemini API キーを取得（優先順: セッション > st.secrets > 環境変数）"""
    # 1. セッションステート（サイドバー入力）
    if "gemini_api_key" in st.session_state and st.session_state["gemini_api_key"]:
        return st.session_state["gemini_api_key"]

    # 2. st.secrets
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 3. 環境変数
    return os.environ.get("GEMINI_API_KEY", "")


def _render_waveform(data: np.ndarray, sr: int, result: CutResult):
    """簡易的な波形プレビューを描画"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 3))

    # 背景色
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # ダウンサンプリング（表示用）
    downsample = max(1, len(data) // 5000)
    plot_data = data[::downsample]
    time_axis = np.linspace(0, len(data) / sr, len(plot_data))

    # 波形描画
    ax.plot(time_axis, plot_data, color="#7850ff", linewidth=0.4, alpha=0.8)
    ax.fill_between(time_axis, plot_data, alpha=0.15, color="#7850ff")

    # 無音区間をハイライト
    for seg in result.silent_segments:
        ax.axvspan(seg.start_sec, seg.end_sec, alpha=0.25, color="#f85149", zorder=0)

    ax.set_xlim(0, len(data) / sr)
    ax.set_ylabel("振幅", color="#8b949e", fontsize=8)
    ax.set_xlabel("時間 (秒)", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=7)

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # 凡例
    sound_patch = mpatches.Patch(color="#7850ff", alpha=0.5, label="有音区間")
    silence_patch = mpatches.Patch(color="#f85149", alpha=0.25, label="無音区間（カット対象）")
    ax.legend(handles=[sound_patch, silence_patch], loc="upper right",
              fontsize=7, facecolor="#0d1117", edgecolor="#30363d",
              labelcolor="#8b949e")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoSRT - 動画編集前処理ツール",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# カスタム CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* --- 全体テーマ --- */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Noto Sans JP', sans-serif;
    }

    /* --- ヘッダー --- */
    .app-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 70%, rgba(120, 80, 255, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 30%, rgba(0, 200, 255, 0.1) 0%, transparent 50%);
        animation: headerShimmer 8s ease-in-out infinite alternate;
    }
    @keyframes headerShimmer {
        from { transform: rotate(0deg); }
        to { transform: rotate(3deg); }
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .app-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
    }

    /* --- メトリクスカード --- */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(120, 80, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(120, 80, 255, 0.5);
        box-shadow: 0 8px 25px rgba(120, 80, 255, 0.15);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7850ff, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* --- 差分表示 --- */
    .diff-container {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'SFMono-Regular', 'Consolas', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    .diff-removed {
        background: rgba(248, 81, 73, 0.15);
        color: #ff7b72;
        padding: 2px 6px;
        border-radius: 3px;
        text-decoration: line-through;
    }
    .diff-added {
        background: rgba(63, 185, 80, 0.15);
        color: #7ee787;
        padding: 2px 6px;
        border-radius: 3px;
    }

    /* --- 波形ビジュアライゼーション --- */
    .viz-section {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* --- サイドバー --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #7850ff;
        font-size: 1.1rem;
    }

    /* --- ステータスバッジ --- */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-success {
        background: rgba(63, 185, 80, 0.15);
        color: #7ee787;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .status-processing {
        background: rgba(120, 80, 255, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(120, 80, 255, 0.3);
    }

    /* --- タブスタイル --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* --- ファイルアップローダー --- */
    .stFileUploader > div {
        border-radius: 12px;
    }

    /* --- プログレスアニメーション --- */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .processing-indicator {
        animation: pulse 2s ease-in-out infinite;
        color: #a78bfa;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# サイドバー（先に描画して API キーが利用可能になるようにする）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ 設定")

    st.markdown("---")
    st.markdown("### 🔑 API キー")
    st.caption("API キーは環境変数または `st.secrets` でも設定可能です。")

    gemini_key_input = st.text_input(
        "Gemini API キー",
        type="password",
        key="gemini_key_input",
        help="Google AI Studio から取得した Gemini API キーを入力してください。",
    )
    if gemini_key_input:
        st.session_state["gemini_api_key"] = gemini_key_input

    st.markdown("---")
    st.markdown("### ℹ️ このアプリについて")
    st.markdown("""
    **AutoSRT** は、動画編集の前処理を自動化する
    Web アプリケーションです。

    **🔇 無音カット機能**
    - 音声ファイルから無音区間を自動検出
    - FCP XML を再構築してカット済タイムラインを生成
    - Premiere Pro でそのまま読み込み可能

    **📝 SRT 字幕校正機能**
    - LLM がコンテキストを考慮して字幕を校正
    - 専門用語・固有名詞の正確な反映
    - タイムスタンプを維持したまま校正
    """)

    st.markdown("---")
    st.caption("© 2026 AutoSRT | Powered by Streamlit")


# ---------------------------------------------------------------------------
# ヘッダー
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🎬 AutoSRT — 動画編集前処理ツール</h1>
    <p>無音カット（FCP XML 生成）と SRT 字幕の AI 校正を、ブラウザ上で簡単に。</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔇 無音カット（XML × Audio）", "📝 SRT 字幕校正（SRT × LLM）"])


# ===========================================================================
# フェーズ1: 無音カット機能
# ===========================================================================
with tab1:
    st.markdown("### 📁 ファイルアップロード")

    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        audio_file = st.file_uploader(
            "🎵 音声ファイル（MP3 / WAV）",
            type=["mp3", "wav"],
            key="audio_upload",
            help="カットの基準となる音声ファイルをアップロードしてください。動画から抽出した音声をご使用ください。",
        )

    with col_upload2:
        xml_file = st.file_uploader(
            "📄 FCP XML ファイル",
            type=["xml"],
            key="xml_upload",
            help="Premiere Pro から書き出した FCP XML ファイルをアップロードしてください。無音区間を除外した新しい XML を生成します。",
        )

    st.markdown("---")
    st.markdown("### ⚙️ パラメータ設定")

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        threshold_pct = st.slider(
            "🔊 音量閾値（最大RMSに対する%）",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            key="threshold",
            help="この値より小さいRMS値の区間を「無音」と判定します。値を大きくするとカットされる区間が増えます。",
        )

    with col_p2:
        min_silence_ms = st.slider(
            "⏱️ 最小無音時間（ミリ秒）",
            min_value=10,
            max_value=3000,
            value=167,
            step=1,
            key="min_silence",
            help="無音と判定する最低持続時間。29.97fps で 5フレーム ≈ 167ms、30fps で 5フレーム ≈ 167ms。",
        )

    with col_p3:
        padding_ms = st.slider(
            "🔲 パディング（ミリ秒）",
            min_value=0,
            max_value=500,
            value=67,
            step=1,
            key="padding",
            help="カット前後に残す余白。29.97fps で 2フレーム ≈ 67ms。発話の頭切れ・尻切れを防ぎます。",
        )

    st.markdown("---")

    # --- 実行ボタン ---
    if st.button("🚀 無音カットを実行", key="run_silence_cut", type="primary", use_container_width=True):
        if audio_file is None:
            st.error("⚠️ 音声ファイルをアップロードしてください。")
        else:
            with st.spinner("🔄 音声解析中... しばらくお待ちください。"):
                try:
                    audio_bytes = audio_file.read()
                    xml_bytes = xml_file.read() if xml_file else None

                    result, new_xml = process_silence_cut(
                        audio_bytes=audio_bytes,
                        audio_filename=audio_file.name,
                        xml_bytes=xml_bytes,
                        threshold_pct=threshold_pct,
                        min_silence_ms=float(min_silence_ms),
                        padding_ms=float(padding_ms),
                    )

                    st.session_state["cut_result"] = result
                    st.session_state["new_xml"] = new_xml

                    st.success("✅ 処理が完了しました！")

                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")

    # --- 結果表示 ---
    if "cut_result" in st.session_state:
        result: CutResult = st.session_state["cut_result"]
        new_xml = st.session_state.get("new_xml")

        st.markdown("### 📊 処理結果")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{_format_time(result.original_duration)}</div>
                <div class="metric-label">元の長さ</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{_format_time(result.cut_duration)}</div>
                <div class="metric-label">カット後の長さ</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            saved = result.original_duration - result.cut_duration
            pct = (saved / result.original_duration * 100) if result.original_duration > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pct:.1f}%</div>
                <div class="metric-label">削減率</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.num_cuts}</div>
                <div class="metric-label">カット箇所数</div>
            </div>
            """, unsafe_allow_html=True)

        # --- セグメント詳細 ---
        with st.expander("📋 セグメント詳細を表示", expanded=False):
            st.markdown("#### 有音区間（残るセグメント）")
            seg_data = []
            for i, seg in enumerate(result.sound_segments):
                seg_data.append({
                    "No.": i + 1,
                    "開始": _format_time(seg.start_sec),
                    "終了": _format_time(seg.end_sec),
                    "長さ": f"{seg.duration:.2f}秒",
                })
            st.dataframe(seg_data, use_container_width=True)

            st.markdown("#### 無音区間（カットされるセグメント）")
            sil_data = []
            for i, seg in enumerate(result.silent_segments):
                sil_data.append({
                    "No.": i + 1,
                    "開始": _format_time(seg.start_sec),
                    "終了": _format_time(seg.end_sec),
                    "長さ": f"{seg.duration:.2f}秒",
                })
            st.dataframe(sil_data, use_container_width=True)

        # --- 波形ビジュアライゼーション ---
        if audio_file is not None:
            with st.expander("📈 波形プレビュー", expanded=False):
                try:
                    audio_file.seek(0)
                    sr, data = load_audio_as_wav(audio_file.read(), audio_file.name)
                    _render_waveform(data, sr, result)
                except Exception:
                    st.warning("波形の表示に失敗しました。")

        # --- ダウンロード ---
        if new_xml is not None:
            st.markdown("### 💾 ダウンロード")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 処理後の FCP XML をダウンロード",
                    data=new_xml,
                    file_name="autosrt_cut.xml",
                    mime="application/xml",
                    type="primary",
                    use_container_width=True,
                )
            with col_dl2:
                if st.button("🖥️ デスクトップに保存", key="save_xml_desktop", use_container_width=True):
                    _save_to_desktop(new_xml, "autosrt_cut.xml")
        elif xml_file is None:
            st.info("💡 FCP XML ファイルもアップロードすると、無音区間を除外した XML をダウンロードできます。")


# ===========================================================================
# フェーズ2: SRT 字幕校正機能
# ===========================================================================
with tab2:
    st.markdown("### 📁 ファイルアップロード")

    srt_file = st.file_uploader(
        "📄 SRT 字幕ファイル",
        type=["srt"],
        key="srt_upload",
        help="校正したい SRT 字幕ファイルをアップロードしてください。",
    )

    st.markdown("---")
    st.markdown("### 🎯 文脈情報の設定")

    col_ctx1, col_ctx2 = st.columns(2)

    with col_ctx1:
        video_context = st.text_area(
            "🎬 動画のジャンルや文脈",
            placeholder="例: 政治に関するインタビュー番組。ゲストは経済学者の山田太郎氏。日本の財政政策についての議論。",
            height=120,
            key="video_context",
            help="動画の内容を詳しく記述してください。LLM がこの情報を参考にして、文脈に合った校正を行います。",
        )

    with col_ctx2:
        terminology = st.text_area(
            "📖 専門用語・固有名詞リスト",
            placeholder="例:\nプライマリーバランス\n量的緩和\nマイナス金利\n日本銀行\n山田太郎",
            height=120,
            key="terminology",
            help="動画内で使われる専門用語や固有名詞を1行に1つずつ入力してください。音声認識の誤変換をこれらの正確な表記に修正します。",
        )

    st.markdown("---")
    st.markdown("### 🤖 LLM 設定")

    col_llm1, col_llm2 = st.columns(2)

    with col_llm1:
        llm_model = st.text_input(
            "モデル名",
            value="gemini-2.0-flash",
            key="llm_model",
            help="使用する Gemini モデル名を入力してください。",
        )

    with col_llm2:
        chunk_size = st.slider(
            "チャンクサイズ（エントリ数）",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            key="chunk_size",
            help="LLM に一度に送信する字幕エントリの数です。大きくすると文脈の理解が良くなりますが、トークン制限に注意してください。",
        )

    # API キーの取得
    api_key = _get_api_key()

    st.markdown("---")

    # --- 実行ボタン ---
    if st.button("🚀 SRT 校正を実行", key="run_srt_correction", type="primary", use_container_width=True):
        if srt_file is None:
            st.error("⚠️ SRT ファイルをアップロードしてください。")
        elif not api_key:
            st.error("⚠️ Gemini API キーが設定されていません。サイドバーまたは環境変数で設定してください。")
        else:
            srt_bytes = srt_file.read()

            progress_bar = st.progress(0, text="📝 SRT 校正を準備中...")

            def update_progress(current, total):
                if total > 0:
                    pct = current / total
                    progress_bar.progress(pct, text=f"📝 チャンク {current}/{total} を処理中...")

            try:
                correction_result = process_srt_correction(
                    srt_bytes=srt_bytes,
                    context=video_context,
                    terminology=terminology,
                    model=llm_model,
                    api_key=api_key,
                    chunk_size=chunk_size,
                    progress_callback=update_progress,
                )

                st.session_state["correction_result"] = correction_result
                progress_bar.progress(1.0, text="✅ 校正が完了しました！")

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")

    # --- 結果表示 ---
    if "correction_result" in st.session_state:
        cr: CorrectionResult = st.session_state["correction_result"]

        st.markdown("### 📊 校正結果")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cr.total_entries}</div>
                <div class="metric-label">字幕エントリ総数</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cr.changed_entries}</div>
                <div class="metric-label">修正されたエントリ数</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r3:
            rate = (cr.changed_entries / cr.total_entries * 100) if cr.total_entries > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rate:.1f}%</div>
                <div class="metric-label">修正率</div>
            </div>
            """, unsafe_allow_html=True)

        # --- 差分プレビュー ---
        st.markdown("### 🔍 校正前後の比較")

        # フィルターオプション
        show_changes_only = st.checkbox("変更のあったエントリのみ表示", value=True, key="show_changes_only")

        diffs_to_show = cr.diffs
        if show_changes_only:
            diffs_to_show = [d for d in cr.diffs if d.has_change]

        if not diffs_to_show:
            st.info("💡 変更はありませんでした。")
        else:
            for diff in diffs_to_show:
                if diff.has_change:
                    with st.container():
                        st.markdown(f"**#{diff.index}**")
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.markdown(f'<div class="diff-container"><span class="diff-removed">{_escape_html(diff.original)}</span></div>', unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(f'<div class="diff-container"><span class="diff-added">{_escape_html(diff.corrected)}</span></div>', unsafe_allow_html=True)
                else:
                    with st.expander(f"#{diff.index} — 変更なし", expanded=False):
                        st.text(diff.original)

        # --- ダウンロード ---
        st.markdown("### 💾 ダウンロード")
        corrected_srt = entries_to_srt(cr.corrected_entries)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 校正後の SRT をダウンロード",
                data=corrected_srt.encode("utf-8"),
                file_name="autosrt_corrected.srt",
                mime="text/plain",
                type="primary",
                use_container_width=True,
            )
        with col_dl2:
            if st.button("🖥️ デスクトップに保存", key="save_srt_desktop", use_container_width=True):
                _save_to_desktop(corrected_srt.encode("utf-8"), "autosrt_corrected.srt")
