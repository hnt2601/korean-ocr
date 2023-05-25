from flask import Flask
from flask_restful import Resource, Api
import requests

app = Flask(__name__)
api = Api(app)
basic_router = "/api/v1"


try:
    from src.controller.ocr import OCR

    api.add_resource(OCR, basic_router + "/kor-ocr")
except Exception as e:
    print("route/__init__.py", e)
    pass
