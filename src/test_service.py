import json
from service import Service
from users import Authentication

service = Service()
general_info = service.get_general_info()
auth = Authentication(username="crfm", password="crfm")
for query in general_info.exampleQueries:
    print(f"=== query: {query}")
    response = service.expand_query(query)
    for request in response.requests:
        print(f"======= request {request}")
        response = service.make_request(auth, request)
        print(response)
        print("")

service.finish()
