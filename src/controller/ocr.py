from flask_restful import Resource, reqparse

from src import server
from src import ocr

import logging

parser = reqparse.RequestParser()
parser.add_argument("image", type=str, location="json")


class OCR(Resource):
    def post(self):
        args = parser.parse_args()
        jpg_as_text = args["image"]
        if jpg_as_text is None:
            server.bad_request(message="Image is None")
        data = ocr(jpg_as_text)
        response = server.response(data=data)
        if not response["data"]:
            return server.bad_request(message="Data is None")
        return response
