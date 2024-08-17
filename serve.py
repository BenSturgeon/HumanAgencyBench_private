from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def list_files():
    files = []
    for root, dirs, filenames in os.walk('cases'):
        for filename in filenames:
            path = os.path.relpath(os.path.join(root, filename), os.getcwd())
            if path.endswith('.html'):
                files.append(path)
    
    html = """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Files in Directory:</h1>
        <ul>
        {% for file in files %}
            <li><a href="{{ file }}">{{ file }}</a></li>
        {% endfor %}
        </ul>
    </body>
    </html>
    """
    return render_template_string(html, files=files)

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(os.getcwd(), filename)

if __name__ == '__main__':
    app.run(debug=True)