class CustomCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization.

    def __call__(self, request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.

        response = self.get_response(request)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Headers"] = "*"

        # Code to be executed for each request/response after
        # the view is called.

        return response


class UserEmailMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Retrieve the user's email from the session or any other storage mechanism
        email = request.session.get('USER_EMAIL', '')  # Adjust this to your actual storage method
        
        # Attach the email to the request object
        request.user_email = email
        
        response = self.get_response(request)
        return response        
