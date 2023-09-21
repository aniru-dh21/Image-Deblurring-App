import cv2
import models
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

class ImageDeblurringApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Image Deblurring App")

        # Set background color and font
        self.window.configure(bg="#E6F1FF")
        self.label_font = ("Times New Roman", 14)
        self.button_font = ("Times New Roman", 12, "bold")

        self.image_file_path = None

        # Create UI elements
        self.image_preview_label = tk.Label(self.window)
        self.image_preview_label.pack(pady=10)

        self.select_image_button = tk.Button(self.window, text="Select Image", font=self.button_font, command=self.choose_image_file)
        self.select_image_button.pack(pady=10)

        deblur_button = tk.Button(self.window, text="Deblur Image", font=self.button_font, command=self.deblur_image)
        deblur_button.pack(pady=10)

        self.deblurred_image_label = tk.Label(self.window)
        self.deblurred_image_label.pack(pady=10)

        self.instructions_label = tk.Label(self.window, text="Instructions:\n\n1. Click the 'Select Image' button to choose an image for deblurring.\n2. Click the 'Deblur Image' button to deblur the selected image.\n\nNote: Please wait for the deblurring process to finish before selecting a new image.",
                                           font=self.label_font, bg="#E6F1FF", anchor="w", justify="left")
        self.instructions_label.pack(pady=10)

    def choose_image_file(self):
        self.image_file_path = filedialog.askopenfilename()
        if self.image_file_path:
            self.instructions_label.pack_forget()

            image_preview = Image.open(self.image_file_path)
            image_preview = image_preview.resize((224, 224), Image.LANCZOS)
            image_preview = ImageTk.PhotoImage(image_preview)
            self.image_preview_label.configure(image=image_preview)
            self.image_preview_label.image = image_preview
            self.deblurred_image_label.configure(image=None)

            self.select_image_button.config(text="Change Image")

    def deblur_image(self):
        if not self.image_file_path:
            return

        # Load the image
        image = cv2.imread(self.image_file_path)
        orig_image = cv2.resize(image, (224, 224))
        cv2.imwrite("original_blurred_image.jpg", orig_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Preprocess the image
        image = transform(image).unsqueeze(0)

        # Load the model
        device = 'cpu'
        model = models.SimpleAE().to(device).eval()
        model.load_state_dict(torch.load('C:/Users/ANIRUDH/OneDrive/Desktop/Image Deblurring App/outputs/model.pth'))

        # Deblur the image
        with torch.no_grad():
            outputs = model(image)
            save_decoded_image(outputs.cpu().data, name="deblurred_image.jpg")
        
        # Show the deblurred image
        deblurred_image = cv2.imread("deblurred_image.jpg")
        deblurred_image = cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB)
        deblurred_image = cv2.resize(deblurred_image, (224, 224))
        deblurred_image = Image.fromarray(deblurred_image)
        deblurred_image = ImageTk.PhotoImage(deblurred_image)
        self.deblurred_image_label.configure(image=deblurred_image)
        self.deblurred_image_label.image = deblurred_image

if __name__ == '__main__':
    window = tk.Tk()
    window.geometry("1200x900")
    app = ImageDeblurringApp(window)
    window.mainloop()
