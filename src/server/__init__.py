def response(code="success", message="", data=None):
    return {"status": code, "data": data}


def success(data=None):
    return response(message="success", data=data)


def bad_request(message="bad request"):
    return response(code=400, message=message)


def internal_server_error():
    return response(code=500, message="internal server error")
