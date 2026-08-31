"""Chess Game Reviewer — run with: streamlit run app.py"""

from __future__ import annotations

import io
import html
import math
import os
import shutil
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Chess Game Review", page_icon="♟", layout="wide")
st.markdown(
    """<style>
    .stApp { background: #f6f7f8; }
    .review-card { background: #20252b; border-radius: 12px; padding: 22px 18px;
                   text-align: center; color: #f8fafc; box-shadow: 0 2px 8px #00000018; }
    .review-label { font-size: 24px; font-weight: 800; margin: 7px 0 12px; }
    .review-eval { font-size: 34px; font-weight: 800; letter-spacing: -1px; }
    .review-caption { color: #aab4bf; font-size: 13px; margin-top: 3px; }
    .summary-card { background:#292e36; border-radius:12px; color:#f8fafc; overflow:hidden; }
    .accuracy-title { text-align:center; padding:10px; font-size:17px; font-weight:750; background:#343a44; }
    .accuracy-values { display:grid; grid-template-columns:1fr 1fr; text-align:center; font-size:25px;
                       padding:10px 0; background:#15191e; }
    .summary-head, .summary-row { display:grid; grid-template-columns:40% 20% 20% 20%; align-items:center;
                                  padding:3px 13px; font-size:14px; }
    .summary-head { padding-top:11px; color:#f8fafc; text-align:center; }
    .summary-row { font-size:15px; }
    .summary-row b { text-align:center; }
    .summary-icon { width:25px; height:25px; border-radius:50%; display:flex; align-items:center;
                    justify-content:center; color:#172033; font-size:13px; font-weight:900; }
    .summary-row .summary-icon { justify-self:center; }
    .player-strip { display:flex; align-items:center; justify-content:space-between; min-height:34px;
                    padding:3px 7px; color:#27313c; }
    .player-name { font-size:18px; font-weight:800; }
    .player-rating { font-size:14px; font-weight:600; color:#64748b; }
    .captured-pieces { color:#48525d; font-size:20px; letter-spacing:-4px; margin-right:9px; }
    .material-edge { color:#617080; font-size:13px; font-weight:750; margin-left:8px; }
    </style>""",
    unsafe_allow_html=True,
)

COLOURS = {
    "Great": "#38bdf8",
    "Best": "#22c55e",
    "Excellent": "#65a30d",
    "Good": "#a3e635",
    "Inaccuracy": "#facc15",
    "Mistake": "#fb923c",
    "Blunder": "#ef4444",
    "Book": "#d6a06f",
}
ICONS = {
    "Great": "!", "Best": "★", "Excellent": "👍", "Good": "✓",
    "Inaccuracy": "?!", "Mistake": "?", "Blunder": "??", "Book": "📖",
}


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


def player_rating(headers: dict, colour: str, fallback: int) -> int:
    """Use the PGN rating when supplied; otherwise use the reviewer's setting."""
    try:
        value = int(headers.get(f"{colour}Elo", fallback))
        return value if 400 <= value <= 3500 else fallback
    except (TypeError, ValueError):
        return fallback


def expected_points(own_evaluation: float, rating: int) -> float:
    """Map an evaluation to a player's expected score (0 to 1).

    Chess.com publishes its expected-points cutoffs but not the underlying
    rating/evaluation model. This is an explicit, adjustable approximation:
    stronger players convert a given evaluation advantage more reliably.
    """
    clamped_eval = max(-12.0, min(12.0, own_evaluation))
    conversion_scale = 2.25 - (max(400, min(3000, rating)) - 400) * 0.00058
    return 1 / (1 + math.exp(-clamped_eval / conversion_scale))


def label_for_expected_loss(loss: float) -> str:
    # Classification bands published by Chess.com. A tiny tolerance treats the
    # engine's exact best move as Best despite floating-point rounding.
    if loss <= 0.002:
        return "Best"
    if loss < 0.02:
        return "Excellent"
    if loss < 0.05:
        return "Good"
    if loss < 0.10:
        return "Inaccuracy"
    if loss < 0.20:
        return "Mistake"
    return "Blunder"


def analyse_game(pgn_text: str, depth: int, fallback_rating: int, progress) -> list[dict]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("I could not find a game in that PGN.")

    moves = list(game.mainline_moves())
    if not moves:
        raise ValueError("The PGN does not contain any moves.")
    white_rating = player_rating(dict(game.headers), "White", fallback_rating)
    black_rating = player_rating(dict(game.headers), "Black", fallback_rating)

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
            root_infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
            if isinstance(root_infos, dict):
                root_infos = [root_infos]
            best_info = root_infos[0]
            played_info = engine.analyse(
                board, chess.engine.Limit(depth=depth), root_moves=[played_move]
            )
            best_move = best_info["pv"][0]
            best_san = board.san(best_move)
            best_eval = white_pawns(best_info["score"])
            played_eval = white_pawns(played_info["score"])
            rating = white_rating if player == "White" else black_rating
            best_points = expected_points(best_eval if player == "White" else -best_eval, rating)
            played_points = expected_points(played_eval if player == "White" else -played_eval, rating)
            points_lost = max(0.0, best_points - played_points)
            label = label_for_expected_loss(points_lost)
            # A public online opening database is no longer reliable for a
            # free unauthenticated app. Use a deterministic local rule instead:
            # in the first eight full moves, an Excellent-or-better move counts
            # as book/theory. This correctly includes openings such as 1.e4 d5.
            book_move = board.ply() < 16 and points_lost < 0.02
            if len(root_infos) > 1:
                second_eval = white_pawns(root_infos[1]["score"])
                second_points = expected_points(second_eval if player == "White" else -second_eval, rating)
                only_good_move = best_points - second_points >= 0.10
            else:
                only_good_move = False
            if book_move:
                label = "Book"
            elif label in {"Best", "Excellent"} and only_good_move:
                label = "Great"
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
                    "expected_points_lost": round(points_lost, 4),
                    "rating_used": rating,
                    "is_book": book_move,
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
    # Draw only the useful coordinates: ranks on the left and files below.
    # python-chess's built-in option draws all four edges, so add our own.
    svg = chess.svg.board(board, size=580, coordinates=False)
    svg = svg.replace('viewBox="0 0 360 360"', 'viewBox="-22 0 382 382"')
    files = "".join(
        f'<text x="{22.5 + 45 * file}" y="376" text-anchor="middle" '
        f'font-family="Arial" font-size="13" font-weight="bold" fill="#475569">{chr(97 + file)}</text>'
        for file in range(8)
    )
    ranks = "".join(
        f'<text x="-11" y="{28 + 45 * row}" text-anchor="middle" '
        f'font-family="Arial" font-size="13" font-weight="bold" fill="#475569">{8 - row}</text>'
        for row in range(8)
    )
    svg = svg.replace("</svg>", files + ranks + "</svg>")
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


def player_strip_html(name: str, rating: str, colour: chess.Color, fen: str) -> str:
    """Render a player name with the opponent's material they have captured."""
    board = chess.Board(fen)
    opponent = not colour
    symbols = {
        chess.PAWN: "♙" if opponent == chess.WHITE else "♟",
        chess.KNIGHT: "♘" if opponent == chess.WHITE else "♞",
        chess.BISHOP: "♗" if opponent == chess.WHITE else "♝",
        chess.ROOK: "♖" if opponent == chess.WHITE else "♜",
        chess.QUEEN: "♕" if opponent == chess.WHITE else "♛",
    }
    starting = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    captured = "".join(
        symbols[piece] * max(0, starting[piece] - len(board.pieces(piece, opponent)))
        for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
    )
    own_lost = sum(values[piece] * max(0, starting[piece] - len(board.pieces(piece, colour))) for piece in starting)
    opponent_lost = sum(values[piece] * max(0, starting[piece] - len(board.pieces(piece, opponent))) for piece in starting)
    material_edge = opponent_lost - own_lost
    edge = f"+{material_edge}" if material_edge > 0 else ""
    return (
        "<div class='player-strip'>"
        f"<span class='player-name'>{html.escape(name)} <span class='player-rating'>{html.escape(rating)}</span></span>"
        f"<span><span class='captured-pieces'>{captured}</span>"
        f"<span class='material-edge'>{edge}</span></span></div>"
    )


def evaluation_chart(rows: list[dict], selected: int) -> alt.Chart:
    chart_data = pd.DataFrame(rows)[["index", "played_eval"]]
    chart_data["zero"] = 0
    base = alt.Chart(chart_data).encode(
        x=alt.X("index:Q", title="Move", axis=alt.Axis(tickCount=8, labelColor="#64748b")),
        y=alt.Y("played_eval:Q", title="Evaluation", scale=alt.Scale(domain=[-6, 6]),
                axis=alt.Axis(values=[-6, -3, 0, 3, 6], labelColor="#64748b")),
    )
    area = base.mark_area(opacity=0.78).encode(
        y2="zero:Q",
        color=alt.condition(alt.datum.played_eval >= 0, alt.value("#d7dde2"), alt.value("#4c5965")),
    )
    line = base.mark_line(color="#20252b", strokeWidth=1.5)
    selected_line = alt.Chart(pd.DataFrame({"index": [selected]})).mark_rule(
        color="#eab308", strokeWidth=2
    ).encode(x="index:Q")
    return (area + line + selected_line).properties(height=210).configure_view(strokeWidth=0).configure_axis(gridColor="#e4e8ec")


def summary_html(rows: list[dict], headers: dict) -> str:
    """Build the compact, two-player review summary displayed beside the board."""
    labels = ["Great", "Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder", "Book"]
    white_rows = [row for row in rows if row["player"] == "White"]
    black_rows = [row for row in rows if row["player"] == "Black"]

    def accuracy(player_rows: list[dict]) -> float:
        if not player_rows:
            return 100.0
        # Compound retained expected points instead of averaging losses. This
        # stops one large blunder in a short game being diluted into a 95% score.
        retained = 1.0
        for row in player_rows:
            retained *= 1 - min(1.0, max(0.0, row["expected_points_lost"]))
        return max(0.0, min(100.0, 100 * retained))

    def count(player_rows: list[dict], label: str) -> int:
        return sum(row["label"] == label for row in player_rows)

    rows_html = "".join(
        f"<div class='summary-row'>"
        f"<span style='color:{COLOURS[label]}'>{label}</span>"
        f"<b style='color:{COLOURS[label]}'>{count(white_rows, label)}</b>"
        f"<span class='summary-icon' style='background:{COLOURS[label]}'>{ICONS[label]}</span>"
        f"<b style='color:{COLOURS[label]}'>{count(black_rows, label)}</b>"
        f"</div>" for label in labels
    )
    return f"""
    <div class='summary-card'>
      <div class='accuracy-title'>Game accuracy</div>
      <div class='accuracy-values'><b>{accuracy(white_rows):.1f}%</b><b>{accuracy(black_rows):.1f}%</b></div>
      <div class='summary-head'><span></span><b>{headers.get('White', 'White')}</b><span></span><b>{headers.get('Black', 'Black')}</b></div>
      {rows_html}
    </div>"""


st.title("♟ Chess Game Review")
st.caption("Paste a PGN, analyse it with Stockfish, then replay the game move by move.")

with st.expander("Paste game PGN", expanded="analysis" not in st.session_state):
    pgn_text = st.text_area("PGN", height=180, placeholder="[Event \"Chess.com game\"]\n\n1. e4 e5 2. Nf3 Nc6 *")
    option_one, option_two = st.columns(2)
    with option_one:
        depth = st.slider("Analysis depth", min_value=10, max_value=20, value=15, help="Higher is slower but more reliable.")
    with option_two:
        fallback_rating = st.number_input("Your rating", min_value=400, max_value=3500, value=1200, step=25,
                                          help="Used only if the PGN has no WhiteElo or BlackElo rating.")
    if st.button("Analyse game", type="primary"):
        if not pgn_text.strip():
            st.error("Paste a PGN first.")
        else:
            try:
                bar = st.progress(0, text="Starting Stockfish…")
                st.session_state.analysis = analyse_game(pgn_text, depth, fallback_rating, bar)
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


def select_move(index: int) -> None:
    st.session_state.selected = index

headers = st.session_state.headers
left, centre, right = st.columns([1.15, 2.5, 1.35])
with left:
    st.altair_chart(evaluation_chart(rows, selected), use_container_width=True)
    st.divider()
    # Keep the move list dense: two move columns and no annotations. The board
    # itself carries the quality sticker for the selected move.
    for first in range(0, len(rows), 2):
        first_column, second_column = st.columns(2, gap="small")
        for column, row in zip((first_column, second_column), rows[first:first + 2]):
            with column:
                st.button(f"{row['move_no']} {row['san']}", key=f"move-{row['index']}", use_container_width=True,
                          on_click=select_move, args=(row["index"],))

with centre:
    st.markdown(player_strip_html(headers.get("Black", "Black"), headers.get("BlackElo", "?"), chess.BLACK, current["fen_after"]), unsafe_allow_html=True)
    st.markdown(board_svg(current["fen_after"], current), unsafe_allow_html=True)
    st.markdown(player_strip_html(headers.get("White", "White"), headers.get("WhiteElo", "?"), chess.WHITE, current["fen_after"]), unsafe_allow_html=True)
    previous, position, following = st.columns([1, 2, 1])
    with previous:
        st.button("← Previous", disabled=selected == 0, use_container_width=True,
                  on_click=select_move, args=(selected - 1,))
    with position:
        st.markdown(f"<p style='text-align:center'><b>{current['move_no']} {current['san']}</b> — {current['label']}</p>", unsafe_allow_html=True)
    with following:
        st.button("Next →", disabled=selected == len(rows) - 1, use_container_width=True,
                  on_click=select_move, args=(selected + 1,))

with right:
    colour = COLOURS[current["label"]]
    st.markdown(
        f"<div class='review-card'>"
        f"<div style='font-size:32px;color:{colour}'>{ICONS[current['label']]}</div>"
        f"<div class='review-label'>{current['label']} move</div>"
        f"<div class='review-eval'>{current['played_eval']:+.2f}</div>"
        f"<div class='review-caption'>evaluation after {current['move_no']} {current['san']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(summary_html(rows, headers), unsafe_allow_html=True)
