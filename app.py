from lumina.core import app
from lumina.routes import register_routes

register_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 7860)), debug=False)
