import json
import os
import platform
import time
from tkinter import BOTH, TOP, Button, Label, Tk, Toplevel, font
import zipfile
from pathlib import Path

import cv2
from jamdict import Jamdict
import numpy as np

import pytesseract
from screeninfo import get_monitors

from sudachipy import tokenizer
from sudachipy import dictionary

import keyboard

from PIL import Image, ImageTk, ImageDraw, ImageGrab, ImageChops

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

if platform.system() == 'Windows':

    pytesseract.pytesseract.tesseract_cmd = str(
        Path(SCRIPT_DIR, 'tesseract', 'tesseract.exe'))


def grab_screen(x, y, x1, y1):
    return ImageGrab.grab(bbox=(x, y, x1, y1))


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def thresh_mask(channel):
    return channel[:, :] < 128


def thresh_apply(image):
    thresh = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    mask = image[:, :] > 128

    thresh[mask] = 255
    return thresh


def clahe_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def idk_contrast(img):
    min = np.min(img)
    max = np.max(img)

    cont = 255 / (max - min) * (img - min)

    print(img.shape)
    print(cont.shape)

    return cont.astype(np.uint8)


def clahe_contrast_gray(img):
    clahe = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def scale_contrast(img):
    return cv2.convertScaleAbs(img, alpha=3, beta=0)


def houghLines(img):
    return cv2.HoughLinesP(img, 1, np.pi / 180, 100)


def scale(img):
    if min(img.shape) < 100:
        return cv2.resize(img, (int(img.shape[1] * 3), int(img.shape[0] * 3)), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def scale2(img):
    if min(img.shape) > 100:
        return cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def trim(im):
    border = 50
    im = Image.fromarray(im)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        cropped = im.crop(bbox)
        bg = Image.new(
            im.mode, (bbox[2] - bbox[0] + border*2, bbox[3] - bbox[1] + border*2), im.getpixel((0, 0)))
        bg.paste(cropped, (border, border))
        return np.array(bg)
    else:
        return np.array(im)


def fill_contour(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(img, [c], 0, (255, 255, 255), -1)

    return img


def line_contour(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    bg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for c in cnts:
        cv2.drawContours(bg, [c], 0, 255, 1)

    return bg


def sharpen(image):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def unsharp_mask(image):
    amount = 5.0
    blurred = cv2.GaussianBlur(image, (3, 3), 5.0)
    return (amount + 1) * image - amount * blurred


def filter_image(image):
    gray = get_grayscale(image)

    cont = idk_contrast(gray)

    if cont[0][0] > 128:
        cont = 255 - cont

    scl = scale(cont)

    thresh = thresh_apply(scl)

    thin = cv2.ximgproc.thinning(thresh)

    dil = dilate(thin)

    trm = trim(dil)

    scl2 = scale2(trm)

    thresh2 = thresholding(scl2)

    thresh2 = 255 - thresh2

    return thresh2


def filter_channels(image):
    b, g, r = cv2.split(image)

    b = thresh_apply(clahe_contrast_gray(b))
    g = thresh_apply(clahe_contrast_gray(g))
    r = thresh_apply(clahe_contrast_gray(r))

    return b, g, r


def filter_text(text):
    text = text.replace('\n', '')
    text = text.replace(' ', '')
    text = text.strip()
    return text


def text_from_channels(image):
    b, g, r = filter_channels(image)

    try:
        if np.shape(image)[0] <= np.shape(image)[1]:
            text_b = pytesseract.image_to_string(
                b, lang="jpn", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 8")
            text_g = pytesseract.image_to_string(
                g, lang="jpn", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 8")
            text_r = pytesseract.image_to_string(
                r, lang="jpn", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 8")
        else:
            text_b = pytesseract.image_to_string(
                b, lang="jpn_vert", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 5")
            text_g = pytesseract.image_to_string(
                g, lang="jpn_vert", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 5")
            text_r = pytesseract.image_to_string(
                r, lang="jpn_vert", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 5")

        return filter_text(text_b), filter_text(text_g), filter_text(text_r)
    except RuntimeError as e:
        print(e)
        return b, g, r, '', '', ''


def text_from_image(image):
    img = filter_image(image)

    if np.shape(image)[0] <= np.shape(image)[1]:
        data = pytesseract.image_to_data(
            img, lang="jpn", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 6", output_type=pytesseract.Output.DICT)
    else:
        data = pytesseract.image_to_data(
            img, lang="jpn_vert", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 5", output_type=pytesseract.Output.DICT)

    print(data)

    text = ""
    start = len(data["conf"]) - data["conf"][::-1].index('-1')
    for i in range(start, len(data["conf"])):
        # to remove duplicate kana
        if data["conf"][i] < np.mean(data["conf"][start:]) * 0.8:
            if len(data["text"][i]) > 1 and data["text"][i][0] == data["text"][i][0]:
                text += data["text"][i][0]
            continue

        text += data["text"][i]

    print(text)

    return filter_text(text)


def lookup_text(text):
    if text == '':
        return ("", "", "")
    look = jam.lookup(text, lookup_chars=False)
    while len(look.entries) == 0 or len(look.entries[0].senses) == 0:
        text = text[:-1]

        if text == '':
            return ("", "", "")

        look = jam.lookup(text, lookup_chars=False)
    return (text, "\n".join([sense.text().replace("/", " / ") for sense in look.entries[0].senses]), look.entries[0].kana_forms[0].text)


def lookup_text_sudachi(text):
    if text == '':
        return ("", "", "")
    if text not in dictionary_map:
        m = tokenizer_obj.tokenize(text, mode)[0]
        text = m.dictionary_form()
        if text not in dictionary_map:
            return lookup_text(text)
    entry = dictionary_map[text][0]
    return (entry[0], "\n".join(entry[5]), entry[1])


def cursor_search(img):
    text = text_from_image(img)

    texts = [lookup_text_sudachi(text), lookup_text(text)]

    texts = [text for text in texts if text[0] != '']

    if len(texts) == 0:
        return ("None", "None", "None")

    texts = sorted(texts, key=lambda x: len(x[0]), reverse=True)

    return (texts[0][0], texts[0][1], texts[0][2])


def load_dictionary(dictionary):
    output_map = {}
    archive = zipfile.ZipFile(dictionary, 'r')

    result = list()
    for file in archive.namelist():
        if file.startswith('term'):
            with archive.open(file) as f:
                data = f.read()
                d = json.loads(data.decode("utf-8"))
                result.extend(d)

    for entry in result:
        if (entry[0] in output_map):
            output_map[entry[0]].append(entry)
        else:
            output_map[entry[0]] = [entry]
    return output_map


class OCR:
    x = y = mx = my = 0
    x1 = y1 = 1
    x_min = y_min = 0
    x_max = y_max = 1

    key_down = False
    mouse_down = False

    select_window = None

    select_label = None
    tooltip_label = None

    tooltip_text = "No text found"

    closed = False

    hotkey = ""

    def __init__(self, root):
        self.root = root

        print("Loading dictionary...")
        self.dictionary_map = load_dictionary(
            str(Path(SCRIPT_DIR, 'dictionaries', 'jmdict_english.zip')))

        print("Initializing defaults for image processing...")
        self.bg_pil = Image.new(
            'RGB', (get_monitors()[0].width, get_monitors()[0].height))
        self.bg_rect = Image.new(
            'RGB', (get_monitors()[0].width, get_monitors()[0].height))
        self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)

        self.button_font = font.Font(
            self.root, family='Arial', size=12, weight='bold')
        self.hotkey_button = Button(self.root, text="Set Activation Hotkey")
        self.hotkey_button.config(font=self.button_font)
        self.hotkey_button.pack(side=TOP, fill=BOTH, expand=True)

        self.hotkey_button.bind("<Button-1>", self.set_hotkey)

        self.root.bind('<Configure>', self.on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_resize(self, event):
        button_width = self.hotkey_button.winfo_width()
        button_height = self.hotkey_button.winfo_height()

        self.button_font['size'] = min(button_width // 16, button_height // 2)

    def on_close(self):
        self.closed = True
        self.root.destroy()
        if self.select_window is not None:
            self.select_window.destroy()
            self.select_window = None

    def set_hotkey(self, event):
        self.hotkey = keyboard.read_hotkey(suppress=False)
        self.hotkey_button.config(text=self.hotkey)

    def order_mouse(self):
        self.x_min = min(self.x, self.x1)
        self.x_max = max(self.x, self.x1)
        self.y_min = min(self.y, self.y1)
        self.y_max = max(self.y, self.y1)

        if self.x_min == self.x_max:
            self.x_max += 1
        if self.y_min == self.y_max:
            self.y_max += 1

    def mouse_button_release(self, event):
        if self.key_down:
            jp, en, pron = cursor_search(np.array(self.bg_pil.crop(
                (self.x_min, self.y_min, self.x_max, self.y_max))))
            if jp != "None":
                self.tooltip_text = jp + "\n" + pron + "\n" + en
            else:
                self.tooltip_text = "No text found"

    def mouse_motion(self, event):
        self.mx, self.my = event.x, event.y

    def mouse_motion_button(self, event):
        if self.key_down:
            self.x1, self.y1 = event.x, event.y
            self.mx, self.my = event.x, event.y

            self.order_mouse()

            self.bg_rect = self.bg_pil.copy()
            img_draw = ImageDraw.Draw(self.bg_rect)
            img_draw.rectangle(
                [self.x_min, self.y_min, self.x_max, self.y_max], outline='red')

    def mouse_button(self, event):
        if self.key_down:
            self.mouse_down = True
            self.x, self.y = event.x, event.y
            self.x1 = self.x + 1
            self.y1 = self.y + 1

            self.order_mouse()

    def main(self):
        print("Initialization complete!")
        while True:
            if self.closed:
                return
            if self.hotkey == "":
                self.root.update()
                continue
            if keyboard.is_pressed(self.hotkey):
                if not self.key_down:
                    self.key_down = True

                    self.bg_pil = grab_screen(
                        0, 0, get_monitors()[0].width, get_monitors()[0].height)
                    self.bg_rect = self.bg_pil.copy()

                    self.select_window = Toplevel()
                    self.select_window.wm_overrideredirect(True)
                    self.select_window.wm_geometry("%dx%d+%d+%d" %
                                                   (get_monitors()[0].width, get_monitors()[0].height, 0, 0))

                    self.select_label = Label(self.select_window)
                    self.select_label.config(image=self.bg_tk)
                    self.select_label.place(
                        x=0, y=0, width=get_monitors()[0].width, height=get_monitors()[0].height)
                    self.select_label.bind("<Button-1>", self.mouse_button)
                    self.select_label.bind(
                        "<B1-Motion>", self.mouse_motion_button)
                    self.select_label.bind(
                        "<Motion>", self.mouse_motion)
                    self.select_label.bind(
                        "<ButtonRelease-1>", self.mouse_button_release)

                    self.tooltip_label = Label(self.select_window, text=self.tooltip_text, justify='left',
                                               background="#ffffff", relief='solid', borderwidth=1,
                                               wraplength=180)
                    self.tooltip_label.place(x=self.mx, y=self.my -
                                             self.tooltip_label.winfo_height() - 10)

                    self.select_window.wm_attributes('-topmost', True)
                self.bg_tk = ImageTk.PhotoImage(image=self.bg_rect)
                self.select_label.config(image=self.bg_tk)

                self.tooltip_label.place(x=self.mx, y=self.my -
                                         self.tooltip_label.winfo_height() - 10)
                self.tooltip_label.config(text=self.tooltip_text)

                self.select_window.update()
            else:
                self.key_down = False
                if self.select_window is not None:
                    self.select_window.destroy()
                    self.select_window = None

            self.root.update()


print("Starting...")
win = Tk()
win.title("Japanese -> English")
win.geometry("%dx%d+%d+%d" %
             (get_monitors()[0].width / 4, get_monitors()[0].width / 32, 0, 0))
win.resizable(False, False)
ocr = OCR(win)

print("Loading jamdict...")
jam = Jamdict()
dictionary_map = {}

print("Loading sudachidict_full...")
tokenizer_obj = dictionary.Dictionary(dict="full").create()
mode = tokenizer.Tokenizer.SplitMode.B

ocr.main()
