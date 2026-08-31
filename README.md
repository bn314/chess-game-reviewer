# Chess Game Reviewer

A small Streamlit chess-review website powered by Stockfish.

## Run locally

Install Stockfish, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push all files to a GitHub repository. In Streamlit Community Cloud, choose `app.py` as the entry point. `packages.txt` makes the service install Stockfish.
