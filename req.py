import requests as r

url = "http://localhost:5000/sms_webhook"
body =  {"From":"5206885500", "Body":"Hi there"}

resp = r.get(url, data = body)

print(resp)