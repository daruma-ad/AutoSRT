"""
srt_processor.py
================
SRT 字幕ファイルの解析と LLM を活用した文脈考慮型校正のロジックモジュール。

主な機能:
- SRT ファイルのパース・書き出し
- テキストのチャンク分割（LLM トークン制限対応）
- LLM（Google Gemini）への校正リクエスト
- 校正結果のマージと差分生成
"""

import re
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import pysrt


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------
@dataclass
class SubtitleEntry:
    """字幕エントリ"""
    index: int
    start: str  # SRT形式のタイムスタンプ "HH:MM:SS,mmm"
    end: str
    text: str


@dataclass
class CorrectionDiff:
    """校正差分"""
    index: int
    original: str
    corrected: str
    has_change: bool


@dataclass
class CorrectionResult:
    """校正結果"""
    original_entries: List[SubtitleEntry]
    corrected_entries: List[SubtitleEntry]
    diffs: List[CorrectionDiff]
    total_entries: int
    changed_entries: int


# ---------------------------------------------------------------------------
# SRT パース / 書き出し
# ---------------------------------------------------------------------------
def parse_srt(srt_bytes: bytes) -> List[SubtitleEntry]:
    """
    SRT ファイルをパースする。
    pysrt のフォールバックとして独自パーサーも用意。
    """
    # BOM 除去
    content = srt_bytes.decode("utf-8-sig", errors="replace")
    # 改行コードを統一（\r\n や \r を \n に変換）
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    entries: List[SubtitleEntry] = []

    try:
        subs = pysrt.from_string(content)
        for sub in subs:
            entries.append(SubtitleEntry(
                index=sub.index,
                start=_pysrt_time_to_str(sub.start),
                end=_pysrt_time_to_str(sub.end),
                text=sub.text,
            ))
    except Exception:
        # pysrt が失敗した場合、自前でパース
        entries = _manual_parse_srt(content)

    return entries


def _pysrt_time_to_str(t) -> str:
    """pysrt の SubRipTime を文字列に変換"""
    return f"{t.hours:02d}:{t.minutes:02d}:{t.seconds:02d},{t.milliseconds:03d}"


def _manual_parse_srt(content: str) -> List[SubtitleEntry]:
    """正規表現ベースの SRT パーサー"""
    # 改行コードを統一（LLM の応答に \r\n が含まれると境界検出が壊れるため）
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"
        r"((?:(?!\n\d+\n\d{2}:\d{2}:\d{2}).)*)",
        re.DOTALL,
    )

    entries = []
    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start = match.group(2).replace(".", ",")
        end = match.group(3).replace(".", ",")
        text = match.group(4).strip()
        entries.append(SubtitleEntry(idx, start, end, text))

    return entries


def entries_to_srt(entries: List[SubtitleEntry]) -> str:
    """SubtitleEntry のリストを SRT 形式の文字列に変換"""
    lines = []
    for i, entry in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{entry.start} --> {entry.end}")
        lines.append(entry.text)
        lines.append("")  # 空行
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 短い字幕の結合
# ---------------------------------------------------------------------------
def merge_short_entries(
    entries: List[SubtitleEntry],
    min_chars: int = 5,
) -> List[SubtitleEntry]:
    """
    指定文字数未満の短い字幕エントリを次のエントリと結合する。

    Parameters
    ----------
    entries : 字幕エントリのリスト
    min_chars : この文字数未満のエントリを結合対象とする

    Returns
    -------
    List[SubtitleEntry] : 結合後のリスト
    """
    if not entries:
        return entries

    merged: List[SubtitleEntry] = []
    i = 0

    while i < len(entries):
        current = entries[i]

        # 現在のエントリが短い場合、次のエントリと結合
        if len(current.text.strip()) < min_chars and i + 1 < len(entries):
            next_entry = entries[i + 1]
            combined = SubtitleEntry(
                index=current.index,
                start=current.start,
                end=next_entry.end,
                text=current.text.strip() + next_entry.text.strip(),
            )
            merged.append(combined)
            i += 2  # 2つ分スキップ
        else:
            merged.append(current)
            i += 1

    # インデックスを振り直す
    for idx, entry in enumerate(merged, 1):
        entry.index = idx

    return merged


# ---------------------------------------------------------------------------
# チャンク分割
# ---------------------------------------------------------------------------
def chunk_entries(
    entries: List[SubtitleEntry],
    chunk_size: int = 30,
) -> List[List[SubtitleEntry]]:
    """
    字幕エントリをチャンクに分割する。
    LLM のコンテキストウィンドウ制限やレート制限に対応。

    Parameters
    ----------
    entries : 字幕エントリのリスト
    chunk_size : 1チャンクあたりのエントリ数

    Returns
    -------
    List[List[SubtitleEntry]] : チャンク化されたリスト
    """
    chunks = []
    for i in range(0, len(entries), chunk_size):
        chunks.append(entries[i:i + chunk_size])
    return chunks


# ---------------------------------------------------------------------------
# LLM 制御
# ---------------------------------------------------------------------------
def build_system_prompt(
    context: str = "",
    terminology: str = "",
    max_chars_per_line: int = 0,
) -> str:
    """LLM に送信するシステムプロンプトを構築"""
    prompt = """あなたは日本語字幕の校正を行うプロフェッショナルな編集者です。
以下のルールに従って、入力された字幕テキストを校正してください。

## 校正ルール
1. **誤字脱字の修正**: 音声認識による誤変換を文脈から推測し、正確な表記に修正してください。
2. **表記ゆれの統一**: 同じ単語・フレーズが異なる表記で現れている場合は統一してください。
3. **句読点の整理**: 不要な句読点の削除・適切な位置への追加を行ってください。
4. **フィラーワードの処理**: 「えー」「あの」「まあ」等は、文脈上不要であれば除去してください。ただし話し手の個性として重要な場合は残してください。
"""

    if max_chars_per_line > 0:
        prompt += f"""
5. **インテリジェントな改行**:
   - 1行が **{max_chars_per_line}文字** を超える場合は、読みやすさを考慮して適切な位置で2行に分けてください。
   - 改行を入れる際は、文節や意味の区切り（助詞「は」「が」「を」「に」の直後、または読点の位置など）を優先してください。
   - 単語の途中（例：「コンピュ」と「ータ」の間）での不自然な改行は絶対に避けてください。
   - **1つの字幕エントリは必ず最大2行まで（改行は1つまで）にしてください。3行以上は絶対に禁止です。**
"""


    else:
        prompt += """
5. **不自然な改行・分割の修正**: 文節の途中で不自然に区切られている場合は、自然な位置で区切り直してください。
"""

    prompt += """
## 重要な制約
- **タイムスタンプは一切変更しないでください。** テキスト部分のみを校正します。
- 各エントリの番号（インデックス）は維持してください。
- 入力されたSRT形式と完全に同じ構造で出力してください。
- 意味を変えたり、内容を要約したりしないでください。あくまで「校正」です。
"""

    if context:
        prompt += f"""
## 動画の文脈情報
この字幕は以下の文脈に関する動画のものです。この情報を参考にして、専門用語や文脈に沿った校正を行ってください。
```
{context}
```
"""

    if terminology:
        prompt += f"""
## 専門用語・固有名詞リスト
以下の用語が動画内で使用される可能性があります。音声認識の誤変換がこれらの用語に該当する場合は、正確な表記に修正してください。
```
{terminology}
```
"""

    return prompt


def build_user_prompt(entries: List[SubtitleEntry]) -> str:
    """LLM に渡すユーザープロンプト（SRT チャンク）を構築"""
    prompt = "以下のSRT字幕テキストを校正してください。出力はSRT形式のままお願いします。\n\n"
    for entry in entries:
        prompt += f"{entry.index}\n"
        prompt += f"{entry.start} --> {entry.end}\n"
        prompt += f"{entry.text}\n\n"
    return prompt


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Google Gemini API を呼び出す。

    Parameters
    ----------
    system_prompt : システムプロンプト（校正ルール等）
    user_prompt : ユーザープロンプト（SRT チャンク）
    model : Gemini モデル名（デフォルト: gemini-2.5-flash）
    api_key : Gemini API キー
    """
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Gemini API キーが設定されていません。サイドバーまたは環境変数 GEMINI_API_KEY で設定してください。")

    genai.configure(api_key=key)

    model_name = model or "gemini-2.5-flash"
    gemini_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
        ),
    )

    response = gemini_model.generate_content(user_prompt)
    return response.text


# ---------------------------------------------------------------------------
# 校正処理
# ---------------------------------------------------------------------------
def parse_llm_response(response: str) -> List[SubtitleEntry]:
    """LLM の返答から SRT エントリをパースする"""
    # 改行コードを統一（LLM が \r\n を返す場合がある）
    cleaned = response.strip().replace("\r\n", "\n").replace("\r", "\n")

    # LLM の応答からコードブロックを除去
    # コードブロックが応答の先頭にない場合も対応
    if "```" in cleaned:
        lines = cleaned.split("\n")
        # ``` で囲まれた範囲を抽出
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("```") and start_idx is None:
                start_idx = i
            elif line.strip() == "```" and start_idx is not None:
                end_idx = i
                break
        if start_idx is not None:
            # 開始行（```srt 等）を除去
            content_lines = lines[start_idx + 1:end_idx] if end_idx else lines[start_idx + 1:]
            cleaned = "\n".join(content_lines)

    return _manual_parse_srt(cleaned)


def compute_diffs(
    original: List[SubtitleEntry],
    corrected: List[SubtitleEntry],
) -> List[CorrectionDiff]:
    """元のエントリと校正後のエントリの差分を計算"""
    diffs = []

    # インデックスでマッチング
    corrected_map = {e.index: e for e in corrected}

    for orig in original:
        corr = corrected_map.get(orig.index)
        if corr:
            has_change = orig.text.strip() != corr.text.strip()
            diffs.append(CorrectionDiff(
                index=orig.index,
                original=orig.text,
                corrected=corr.text,
                has_change=has_change,
            ))
        else:
            # 校正後に該当エントリが見つからない場合はそのまま
            diffs.append(CorrectionDiff(
                index=orig.index,
                original=orig.text,
                corrected=orig.text,
                has_change=False,
            ))

    return diffs


def process_srt_correction(
    srt_bytes: bytes,
    context: str = "",
    terminology: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_size: int = 30,
    max_chars_per_line: int = 0,
    min_merge_chars: int = 0,
    progress_callback=None,
) -> CorrectionResult:
    """
    SRT 校正処理のメインロジック。

    Parameters
    ----------
    srt_bytes : SRT ファイルのバイトデータ
    context : 動画の文脈情報
    terminology : 専門用語リスト
    model : LLM モデル名
    api_key : API キー
    chunk_size : チャンクサイズ
    max_chars_per_line : 1行あたりの最大文字数（0の場合は改行なし）
    min_merge_chars : この文字数未満の短い字幕を次と結合（0の場合は結合なし）
    progress_callback : 進捗コールバック関数 (current, total) -> None

    Returns
    -------
    CorrectionResult : 校正結果
    """
    # SRT パース
    original_entries = parse_srt(srt_bytes)

    if not original_entries:
        raise ValueError("SRT ファイルに字幕エントリが見つかりませんでした。")

    # 短い字幕の結合（前処理）
    if min_merge_chars > 0:
        original_entries = merge_short_entries(original_entries, min_merge_chars)

    # チャンク分割
    chunks = chunk_entries(original_entries, chunk_size)

    # システムプロンプト構築
    system_prompt = build_system_prompt(context, terminology, max_chars_per_line)

    # チャンクごとに LLM 呼び出し
    corrected_entries: List[SubtitleEntry] = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))

        user_prompt = build_user_prompt(chunk)

        try:
            response = call_llm(
                system_prompt, user_prompt,
                model=model,
                api_key=api_key,
            )
            parsed = parse_llm_response(response)

            if parsed:
                # タイムスタンプは元のものを使用（位置ベースで強制復元）
                # AIがインデックス番号を変更しても、順番で確実にマッチさせる
                for j, entry in enumerate(parsed):
                    if j < len(chunk):
                        entry.start = chunk[j].start
                        entry.end = chunk[j].end
                        entry.index = chunk[j].index
                    # リテラル "\n" を実際の改行に変換
                    entry.text = entry.text.replace("\\n", "\n")
                    # 3行以上防止: 改行が2つ以上ある場合は最初の1つだけ残す
                    text_lines = entry.text.split("\n")
                    if len(text_lines) > 2:
                        entry.text = text_lines[0] + "\n" + "".join(text_lines[1:])
                # AIが返したエントリ数が元と違う場合、元のエントリ数に合わせる
                if len(parsed) < len(chunk):
                    # 校正済みエントリを先に追加（順序を維持）
                    corrected_entries.extend(parsed)
                    # 不足分は元のエントリをそのまま末尾に追加
                    for k in range(len(parsed), len(chunk)):
                        corrected_entries.append(chunk[k])
                elif len(parsed) > len(chunk):
                    # 超過分は無視（元のチャンク数に合わせる）
                    corrected_entries.extend(parsed[:len(chunk)])
                else:
                    corrected_entries.extend(parsed)
            else:
                # パース失敗時は元のエントリをそのまま使用
                corrected_entries.extend(chunk)

        except Exception as e:
            # エラー時は元のエントリをそのまま使用
            corrected_entries.extend(chunk)
            if progress_callback:
                # エラー情報を含めて通知（progressコールバックに追加情報は渡せないが、
                # 例外をログに記録する処理は呼び出し元で行う）
                pass

    if progress_callback:
        progress_callback(len(chunks), len(chunks))

    # 差分計算
    diffs = compute_diffs(original_entries, corrected_entries)

    return CorrectionResult(
        original_entries=original_entries,
        corrected_entries=corrected_entries,
        diffs=diffs,
        total_entries=len(original_entries),
        changed_entries=sum(1 for d in diffs if d.has_change),
    )
