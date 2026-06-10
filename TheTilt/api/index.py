import os

from flask import Flask, jsonify, send_from_directory

# /api/search and its CORS pin were removed in issue #197: no page ever
# called the endpoint (all search is client-side via search_index.json in
# common.js initGlobalSearch), it matched only the short `player` field, and
# the CORS origin pinned to the deployment subdomain would have silently
# broken any future custom domain and all preview deployments.
app = Flask(__name__, static_folder="../public", static_url_path="")


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api")
def api_info():
    return jsonify({
        "app": "The TILT API",
        "version": "1.0.0",
        "description": "Win Probability Added for IPL Cricket",
        "endpoints": {},
    })


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", port=5002)
