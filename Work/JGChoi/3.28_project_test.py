import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests
import io

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('OCR Image Cropper')
        self.geometry("1000x800")

        # 이미지를 로드하는 버튼
        load_button = tk.Button(self, text="Load Image", command=self.load_image)
        load_button.pack()

        # OCR 실행 버튼
        ocr_button = tk.Button(self, text="OCR", command=self.do_ocr)
        ocr_button.pack()

        # Canvas 생성
        self.canvas = tk.Canvas(self, cursor="cross", width=600, height=400)
        self.canvas.pack(fill="both", expand=True)

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.image = None
        self.tk_image = None
        self.image_path = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if not self.image_path:
            return
        self.image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_drag(self, event):
        curX, curY = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def do_ocr(self):
        if self.image_path and self.rect:
            api_key = "YOUR_API_KEY"
            url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
            
            # 이미지를 메모리에 로드합니다.
            with open(self.image_path, 'rb') as image_file:
                content = image_file.read()
            image_content = base64.b64encode(content).decode()

            # API 요청 본문을 구성합니다.
            body = {
                "requests": [{
                    "image": {"content": image_content},
                    "features": [{"type": "TEXT_DETECTION"}]
                }]
            }
            
            # 요청을 전송합니다.
            response = requests.post(url, json=body)
            result = response.json()
            
            # OCR 결과를 출력합니다.
            text_annotations = result.get('responses', [])[0].get('textAnnotations', [])
            if text_annotations:
                text = text_annotations[0].get('description', '')
                print("OCR 결과:")
                print(text)
            else:
                print("텍스트를 찾을 수 없습니다.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
