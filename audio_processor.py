"""
audio_processor.py
==================
音声解析と FCP XML 再構築のロジックモジュール。

主な機能:
- WAV/MP3 音声ファイルを NumPy 配列として読み込み
- RMS ベースの無音区間検出
- FCP XML のタイムライン解析・再構築
- Premiere Pro 互換の ppro Ticks 計算
"""

import io
import subprocess
import tempfile
import copy
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from lxml import etree


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
PPRO_TICKS_PER_SECOND = 254016000000  # Premiere Pro のティックレート


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------
@dataclass
class SilentSegment:
    """無音区間を表す"""
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class SoundSegment:
    """有音区間を表す"""
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class CutResult:
    """カット処理結果"""
    original_duration: float
    cut_duration: float
    num_cuts: int
    sound_segments: List[SoundSegment]
    silent_segments: List[SilentSegment]


# ---------------------------------------------------------------------------
# 音声読み込み
# ---------------------------------------------------------------------------
def load_audio_as_wav(audio_bytes: bytes, filename: str) -> Tuple[int, np.ndarray]:
    """
    音声ファイル（MP3/WAV）を読み込み、(サンプルレート, numpy配列) を返す。
    MP3 の場合は ffmpeg で WAV に変換してから読み込む。
    """
    suffix = filename.lower().split(".")[-1]

    if suffix == "wav":
        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        # ステレオの場合はモノラルに変換
        if data.ndim > 1:
            data = data.mean(axis=1)
        # float32 に正規化
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        return sr, data

    # MP3 等の場合は ffmpeg で WAV に変換
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_out:
            cmd = [
                "ffmpeg", "-y", "-i", tmp_in.name,
                "-ac", "1", "-ar", "16000", "-f", "wav",
                tmp_out.name,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            sr, data = wavfile.read(tmp_out.name)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            return sr, data


# ---------------------------------------------------------------------------
# 無音検出
# ---------------------------------------------------------------------------
def compute_rms(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    スライディングウィンドウで RMS を計算する。
    """
    # window_size サンプルごとに RMS を計算
    n_windows = len(data) // window_size
    if n_windows == 0:
        return np.array([np.sqrt(np.mean(data ** 2))])
    rms = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = data[i * window_size: (i + 1) * window_size]
        rms[i] = np.sqrt(np.mean(chunk ** 2))
    return rms


def detect_silence(
    data: np.ndarray,
    sample_rate: int,
    threshold_pct: float = 5.0,
    min_silence_ms: float = 500.0,
    window_ms: float = 50.0,
) -> List[SilentSegment]:
    """
    RMS ベースで無音区間を検出する。

    Parameters
    ----------
    data : 音声データ (float32, モノラル)
    sample_rate : サンプルレート
    threshold_pct : 最大 RMS に対する閾値パーセンテージ (0-100)
    min_silence_ms : 無音と判定する最小持続時間 (ミリ秒)
    window_ms : RMS 計算のウィンドウサイズ (ミリ秒)

    Returns
    -------
    List[SilentSegment] : 検出された無音区間のリスト
    """
    window_size = int(sample_rate * window_ms / 1000.0)
    if window_size < 1:
        window_size = 1

    rms = compute_rms(data, window_size)
    max_rms = np.max(rms) if len(rms) > 0 else 1.0
    if max_rms == 0:
        max_rms = 1.0

    threshold = max_rms * (threshold_pct / 100.0)
    is_silent = rms < threshold

    # 無音区間を連続領域にまとめる
    segments: List[SilentSegment] = []
    in_silence = False
    start_idx = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start_idx = i
            in_silence = True
        elif not silent and in_silence:
            start_sec = start_idx * window_ms / 1000.0
            end_sec = i * window_ms / 1000.0
            segments.append(SilentSegment(start_sec, end_sec))
            in_silence = False

    # 末尾が無音のまま終了した場合
    if in_silence:
        start_sec = start_idx * window_ms / 1000.0
        end_sec = len(is_silent) * window_ms / 1000.0
        segments.append(SilentSegment(start_sec, end_sec))

    # 最小無音時間でフィルタリング
    min_silence_sec = min_silence_ms / 1000.0
    segments = [s for s in segments if s.duration >= min_silence_sec]

    return segments


def get_sound_segments(
    total_duration: float,
    silent_segments: List[SilentSegment],
    padding_ms: float = 100.0,
) -> List[SoundSegment]:
    """
    無音区間から有音区間（残すべきセグメント）を算出する。
    パディングを考慮して、無音区間の前後に余白を残す。

    Parameters
    ----------
    total_duration : 音声全体の長さ（秒）
    silent_segments : 検出された無音区間
    padding_ms : 前後に残す余白（ミリ秒）

    Returns
    -------
    List[SoundSegment] : 有音区間のリスト
    """
    if not silent_segments:
        return [SoundSegment(0, total_duration)]

    padding_sec = padding_ms / 1000.0
    sound_segments: List[SoundSegment] = []

    # 無音区間をパディング分縮小してから反転
    adjusted_silences = []
    for seg in silent_segments:
        adj_start = seg.start_sec + padding_sec
        adj_end = seg.end_sec - padding_sec
        if adj_start < adj_end:
            adjusted_silences.append(SilentSegment(adj_start, adj_end))

    if not adjusted_silences:
        return [SoundSegment(0, total_duration)]

    # 最初の有音区間
    if adjusted_silences[0].start_sec > 0:
        sound_segments.append(SoundSegment(0, adjusted_silences[0].start_sec))

    # 無音区間の間の有音区間
    for i in range(len(adjusted_silences) - 1):
        s_start = adjusted_silences[i].end_sec
        s_end = adjusted_silences[i + 1].start_sec
        if s_start < s_end:
            sound_segments.append(SoundSegment(s_start, s_end))

    # 最後の有音区間
    if adjusted_silences[-1].end_sec < total_duration:
        sound_segments.append(SoundSegment(adjusted_silences[-1].end_sec, total_duration))

    return sound_segments


# ---------------------------------------------------------------------------
# FCP XML 処理
# ---------------------------------------------------------------------------
def sec_to_ppro_ticks(seconds: float) -> int:
    """秒を Premiere Pro のティック値に変換"""
    return int(round(seconds * PPRO_TICKS_PER_SECOND))


def sec_to_timebase_frames(seconds: float, timebase: int, ntsc: bool = False) -> int:
    """
    秒をフレーム数に変換。
    NTSC の場合は実際の FPS（例: 29.97）を使い、公称値（30）との
    累積ドリフトを防止する。
    """
    if ntsc:
        actual_fps = timebase * 1000.0 / 1001.0
        return int(round(seconds * actual_fps))
    return int(round(seconds * timebase))


def parse_fcp_xml(xml_bytes: bytes) -> etree._Element:
    """FCP XML をパースし、ルート要素を返す"""
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.fromstring(xml_bytes, parser)
    return tree


def get_timebase(root: etree._Element) -> int:
    """XML からタイムベース（フレームレート）を取得"""
    # sequence > rate > timebase を探す
    tb_elem = root.find(".//sequence/rate/timebase")
    if tb_elem is not None and tb_elem.text:
        return int(tb_elem.text)
    # 見つからない場合は一般的な値を返す
    tb_elem = root.find(".//rate/timebase")
    if tb_elem is not None and tb_elem.text:
        return int(tb_elem.text)
    return 30  # デフォルト


def get_ntsc(root: etree._Element) -> bool:
    """XML から NTSC フラグを取得"""
    ntsc_elem = root.find(".//sequence/rate/ntsc")
    if ntsc_elem is not None and ntsc_elem.text:
        return ntsc_elem.text.upper() == "TRUE"
    ntsc_elem = root.find(".//rate/ntsc")
    if ntsc_elem is not None and ntsc_elem.text:
        return ntsc_elem.text.upper() == "TRUE"
    return False


def get_actual_fps(timebase: int, ntsc: bool) -> float:
    """実際の FPS を計算"""
    if ntsc:
        return timebase * 1000.0 / 1001.0
    return float(timebase)


def rebuild_fcp_xml(
    xml_bytes: bytes,
    sound_segments: List[SoundSegment],
) -> bytes:
    """
    FCP XML を、有音区間のみ残すように再構築する。

    元の XML 構造を尊重し、clipitem の in/out/start/end を
    有音区間に基づいて再計算する。

    NTSC (29.97fps 等) 環境では実際の FPS でフレーム計算を行い、
    pproTicks もフレーム境界に揃えることで累積ドリフトを防止する。
    """
    root = parse_fcp_xml(xml_bytes)
    timebase = get_timebase(root)
    ntsc = get_ntsc(root)
    actual_fps = get_actual_fps(timebase, ntsc)

    # sequence 内の video > track を取得
    tracks = root.findall(".//sequence/media/video/track")
    audio_tracks = root.findall(".//sequence/media/audio/track")

    all_tracks = list(tracks) + list(audio_tracks)

    for track in all_tracks:
        clipitems = track.findall("clipitem")
        if not clipitems:
            continue

        # 元の clipitem をテンプレートとして使用
        template_clip = copy.deepcopy(clipitems[0])

        # 既存の clipitem を全削除
        for ci in clipitems:
            track.remove(ci)

        # 有音区間ごとに clipitem を作成
        timeline_pos = 0  # タイムライン上の現在位置（フレーム）

        for idx, seg in enumerate(sound_segments):
            new_clip = copy.deepcopy(template_clip)

            # フレーム数計算（NTSC 対応: 実際の FPS を使用）
            in_frame = sec_to_timebase_frames(seg.start_sec, timebase, ntsc)
            out_frame = sec_to_timebase_frames(seg.end_sec, timebase, ntsc)
            duration_frames = out_frame - in_frame

            start_frame = timeline_pos
            end_frame = timeline_pos + duration_frames

            # clipitem の各要素を更新
            _set_xml_text(new_clip, "name", f"Segment_{idx + 1:03d}")

            # id 属性を更新
            id_elem = new_clip.get("id")
            if id_elem:
                new_clip.set("id", f"clipitem-{idx + 1}")

            _set_xml_text(new_clip, "in", str(in_frame))
            _set_xml_text(new_clip, "out", str(out_frame))
            _set_xml_text(new_clip, "start", str(start_frame))
            _set_xml_text(new_clip, "end", str(end_frame))

            # pproTicks の更新
            # フレーム境界に揃えた秒数から計算し、フレーム番号との一貫性を保証
            frame_aligned_in_sec = in_frame / actual_fps
            frame_aligned_out_sec = out_frame / actual_fps
            _set_xml_text(
                new_clip, "pproTicksIn",
                str(sec_to_ppro_ticks(frame_aligned_in_sec))
            )
            _set_xml_text(
                new_clip, "pproTicksOut",
                str(sec_to_ppro_ticks(frame_aligned_out_sec))
            )

            track.append(new_clip)
            timeline_pos = end_frame

    # sequence の duration を更新（NTSC 対応）
    total_sound_duration = sum(s.duration for s in sound_segments)
    seq_dur_elem = root.find(".//sequence/duration")
    if seq_dur_elem is not None:
        seq_dur_elem.text = str(sec_to_timebase_frames(total_sound_duration, timebase, ntsc))

    # XML をバイト列として出力
    result = etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
    return result


def _set_xml_text(parent: etree._Element, tag: str, value: str):
    """XML 要素のテキストを設定。要素がなければ作成する。"""
    elem = parent.find(tag)
    if elem is not None:
        elem.text = value
    else:
        elem = etree.SubElement(parent, tag)
        elem.text = value


# ---------------------------------------------------------------------------
# メインの処理関数
# ---------------------------------------------------------------------------
def process_silence_cut(
    audio_bytes: bytes,
    audio_filename: str,
    xml_bytes: Optional[bytes],
    threshold_pct: float = 5.0,
    min_silence_ms: float = 500.0,
    padding_ms: float = 100.0,
) -> Tuple[CutResult, Optional[bytes]]:
    """
    無音カット処理のメイン関数。

    Parameters
    ----------
    audio_bytes : 音声ファイルのバイトデータ
    audio_filename : 音声ファイル名
    xml_bytes : FCP XML のバイトデータ（None の場合は XML 再構築をスキップ）
    threshold_pct : 音量閾値（最大RMSに対するパーセンテージ）
    min_silence_ms : 最小無音時間（ミリ秒）
    padding_ms : パディング（ミリ秒）

    Returns
    -------
    (CutResult, Optional[bytes]) : 処理結果と再構築された XML
    """
    # 音声読み込み
    sr, data = load_audio_as_wav(audio_bytes, audio_filename)
    total_duration = len(data) / sr

    # 無音検出
    silent_segs = detect_silence(
        data, sr,
        threshold_pct=threshold_pct,
        min_silence_ms=min_silence_ms,
    )

    # 有音区間算出
    sound_segs = get_sound_segments(total_duration, silent_segs, padding_ms)

    # カット後の総時間
    cut_duration = sum(s.duration for s in sound_segs)

    result = CutResult(
        original_duration=total_duration,
        cut_duration=cut_duration,
        num_cuts=len(silent_segs),
        sound_segments=sound_segs,
        silent_segments=silent_segs,
    )

    # XML 再構築
    new_xml = None
    if xml_bytes is not None:
        new_xml = rebuild_fcp_xml(xml_bytes, sound_segs)

    return result, new_xml
