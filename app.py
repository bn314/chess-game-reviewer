"""Chess Game Reviewer — run with: streamlit run app.py"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Chess Game Review", page_icon="♟", layout="wide")

COLOURS = {
    "Best": "#22c55e",
    "Excellent": "#65a30d",
    "Good": "#a3e635",
    "Inaccuracy": "#facc15",
    "Mistake": "#fb923c",
    "Blunder": "#ef4444",
}
ICONS = {"Best": "★", "Excellent": "✓", "Good": "✓", "Inaccuracy": "?!", "Mistake": "?", "Blunder": "??"}


def stockfish_path() -> str:
    candidates = [
        os.getenv("STOCKFISH_PATH"),
        "/usr/games/stockfish",  # Debian / Streamlit Community Cloud package
        "/usr/bin/stockfish",
        shutil.which("stockfish"),
        shutil.which("stockfish.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Stockfish is not installed. For Streamlit Community Cloud, keep packages.txt in the repository."
    )


def white_pawns(score: chess.engine.PovScore) -> float:
    """A score in pawns from White's perspective; mate is capped for charting."""
    return score.white().score(mate_score=10_000) / 100


def label_for_loss(loss: float) -> str:
    if loss < 0.10:
        return "Best"
    if loss < 0.35:
        return "Excellent"
    if loss < 0.75:
        return "Good"
    if loss < 1.50:
        return "Inaccuracy"
    if loss < 3.00:
        return "Mistake"
    return "Blunder"


def analyse_game(pgn_text: str, depth: int, progress) -> list[dict]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("I could not find a game in that PGN.")

    moves = list(game.mainline_moves())
    if not moves:
        raise ValueError("The PGN does not contain any moves.")

    board = game.board()
    results: list[dict] = []
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    try:
        for index, played_move in enumerate(moves):
            player = "White" if board.turn else "Black"
            move_number = board.fullmove_number
            san = board.san(played_move)
            position_fen = board.fen()

            # Both scores come from the SAME root position at the SAME depth.
            # This prevents a best move being marked inaccurate merely because a
            # second search looked one ply further ahead.
            best_info = engine.analyse(board, chess.engine.Limit(depth=depth))
            played_info = engine.analyse(
                board, chess.engine.Limit(depth=depth), root_moves=[played_move]
            )
            best_move = best_info["pv"][0]
            best_san = board.san(best_move)
            best_eval = white_pawns(best_info["score"])
            played_eval = white_pawns(played_info["score"])

            raw_loss = best_eval - played_eval if player == "White" else played_eval - best_eval
            loss = max(0.0, raw_loss)
            label = label_for_loss(loss)
            board.push(played_move)
            results.append(
                {
                    "index": index,
                    "move_no": f"{move_number}." if player == "White" else f"{move_number}...",
                    "player": player,
                    "san": san,
                    "uci": played_move.uci(),
                    "to_square": chess.square_name(played_move.to_square),
                    "best_san": best_san,
                    "best_eval": round(best_eval, 2),
                    "played_eval": round(played_eval, 2),
                    "loss": round(loss, 2),
                    "label": label,
                    "fen_before": position_fen,
                    "fen_after": board.fen(),
                }
            )
            progress.progress((index + 1) / len(moves), text=f"Analysing move {index + 1} of {len(moves)}")
    finally:
        engine.quit()
    return results


def board_svg(fen: str, last_move: dict | None) -> str:
    board = chess.Board(fen)
    svg = chess.svg.board(board, size=620, coordinates=True)
    if last_move is None:
        return svg
    square = chess.parse_square(last_move["to_square"])
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    # python-chess board SVG has a 45px-square, 390px viewBox. Draw a small
    # Chess.com-style quality sticker in the top-right of the destination square.
    x, y = file_index * 45 + 36, (7 - rank_index) * 45 + 9
    colour, icon = COLOURS[last_move["label"]], ICONS[last_move["label"]]
    sticker = (
        f'<circle cx="{x}" cy="{y}" r="8.5" fill="{colour}" stroke="#ffffff" stroke-width="1.5"/>'
        f'<text x="{x}" y="{y + 3}" text-anchor="middle" font-family="Arial" '
        f'font-size="7" font-weight="bold" fill="#172033">{icon}</text>'
    )
    return svg.replace("</svg>", sticker + "</svg>")


def move_button(row: dict, selected: bool) -> None:
    colour = COLOURS[row["label"]]
    marker = f"<span style='color:{colour};font-weight:800'>{ICONS[row['label']]}</span>"
    active = "background:#e8f0fe;" if selected else ""
    st.markdown(
        f"<div style='{active}padding:5px 8px;border-radius:6px;margin:2px 0'>"
        f"{marker} <b>{row['move_no']} {row['san']}</b> <small>({row['label']})</small></div>",
        unsafe_allow_html=True,
    )


st.title("♟ Chess Game Review")
st.caption("Paste a PGN, analyse it with Stockfish, then replay the game move by move.")

with st.expander("Paste game PGN", expanded="analysis" not in st.session_state):
    pgn_text = st.text_area("PGN", height=180, placeholder="[Event \"Chess.com game\"]\n\n1. e4 e5 2. Nf3 Nc6 *")
    depth = st.slider("Analysis depth", min_value=10, max_value=20, value=15, help="Higher is slower but more reliable.")
    if st.button("Analyse game", type="primary"):
        if not pgn_text.strip():
            st.error("Paste a PGN first.")
        else:
            try:
                bar = st.progress(0, text="Starting Stockfish…")
                st.session_state.analysis = analyse_game(pgn_text, depth, bar)
                st.session_state.selected = 0
                st.session_state.headers = dict(chess.pgn.read_game(io.StringIO(pgn_text)).headers)
                bar.empty()
            except Exception as error:
                st.error(str(error))

if "analysis" not in st.session_state:
    st.info("Paste a PGN above and select Analyse game to start.")
    st.stop()

rows: list[dict] = st.session_state.analysis
selected = st.session_state.selected
selected = max(0, min(selected, len(rows) - 1))
st.session_state.selected = selected
current = rows[selected]

left, centre, right = st.columns([1.15, 2.5, 1.35])
with left:
    headers = st.session_state.headers
    st.subheader(f"{headers.get('White', 'White')} vs {headers.get('Black', 'Black')}")
    st.caption(f"Result: {headers.get('Result', '*')}  •  {len(rows)} half-moves")
    st.divider()
    for row in rows:
        if st.button(f"{row['move_no']} {row['san']}  {ICONS[row['label']]}", key=f"move-{row['index']}", use_container_width=True):
            st.session_state.selected = row["index"]
            st.rerun()

with centre:
    st.markdown(board_svg(current["fen_after"], current), unsafe_allow_html=True)
    previous, position, following = st.columns([1, 2, 1])
    with previous:
        if st.button("← Previous", disabled=selected == 0, use_container_width=True):
            st.session_state.selected -= 1
            st.rerun()
    with position:
        st.markdown(f"<p style='text-align:center'><b>{current['move_no']} {current['san']}</b> — {current['label']}</p>", unsafe_allow_html=True)
    with following:
        if st.button("Next →", disabled=selected == len(rows) - 1, use_container_width=True):
            st.session_state.selected += 1
            st.rerun()

with right:
    st.subheader(current["label"])
    st.metric("Estimated loss", f"{current['loss']:.2f} pawns")
    st.write(f"**Played:** {current['san']}")
    st.write(f"**Stockfish preferred:** {current['best_san']}")
    st.caption(f"Best evaluation: {current['best_eval']:+.2f} • after played move: {current['played_eval']:+.2f}")
    st.divider()
    chart = pd.DataFrame(rows)[["index", "played_eval"]].set_index("index")
    st.caption("Evaluation after each move (positive = White advantage)")
    st.line_chart(chart, height=160)
    st.divider()
    st.caption("This is an engine-based first version. ‘Brilliant’ and ‘Great Find’ need separate tactical rules and will be added later.")
