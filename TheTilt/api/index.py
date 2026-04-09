import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="../public", static_url_path="")
CORS(app)

DATA_DIR = Path(__file__).parent.parent / "public" / "data"


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api")
def api_info():
    return jsonify({
        "app": "The TILT API",
        "version": "1.0.0",
        "description": "Win Probability Added for IPL Cricket",
        "endpoints": {
            "/api/search?q=<query>": "Search players by name",
        },
    })


@app.route("/api/search")
def search_players():
    query = request.args.get("q", "").lower().strip()
    if not query:
        return jsonify([])

    rankings_path = DATA_DIR / "tilt_rankings.json"
    if not rankings_path.exists():
        return jsonify({"error": "Rankings data not yet generated"}), 404

    with open(rankings_path) as f:
        rankings = json.load(f)

    results = [p for p in rankings if query in p["player"].lower()]
    return jsonify(results[:20])


if __name__ == "__main__":
    app.run(debug=True, port=5002)
