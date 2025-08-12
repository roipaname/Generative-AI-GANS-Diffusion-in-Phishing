import requests

BASE_URL = "http://localhost:8000"

def test_generate_text():
    resp = requests.post(f"{BASE_URL}/generate-text", data={"prompt": "Phishing emails are"})
    print("Generate text response:", resp.json())

def test_classify_text():
    resp = requests.post(f"{BASE_URL}/classify-text", data={"text": "You won a free prize! Click here."})
    print("Classify text response:", resp.json())

def test_classify_image():
    with open("test_phishing_image.jpg", "rb") as f:
        files = {"file": ("test_phishing_image.jpg", f, "image/jpeg")}
        resp = requests.post(f"{BASE_URL}/classify-image", files=files)
        print("Classify image response:", resp.json())

def test_classify_text_image():
    with open("test_phishing_image.jpg", "rb") as f:
        files = {"file": ("test_phishing_image.jpg", f, "image/jpeg")}
        data = {"text": "This is a phishing attempt."}
        resp = requests.post(f"{BASE_URL}/classify-text-image", data=data, files=files)
        print("Classify text+image response:", resp.json())

def test_generate_image():
    resp = requests.post(f"{BASE_URL}/generate-image", data={"prompt": "A futuristic cityscape"})
    print("Generate image response:", resp.json())

if __name__ == "__main__":
    test_generate_text()
    test_classify_text()
    test_classify_image()
    test_classify_text_image()
    test_generate_image()
