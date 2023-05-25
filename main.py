from src.route import app
from src.config import HOST, PORT

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
