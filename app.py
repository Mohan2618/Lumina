from flask import request
from lumina.core import app
from lumina.routes import register_routes

register_routes(app)


@app.after_request
def inject_response_enhancements(response):
    """Load the structured-response UI without changing the existing template."""
    content_type = response.headers.get("Content-Type", "")
    if response.status_code == 200 and content_type.startswith("text/html"):
        body = response.get_data(as_text=True)
        marker = '<script src="/static/response_enhancements.js?v=1"></script>'
        if marker not in body and "</body>" in body:
            body = body.replace("</body>", f"{marker}</body>")
            response.set_data(body)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 7860)), debug=False)
