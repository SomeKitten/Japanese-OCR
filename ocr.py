import _thread
import ctypes
import json
import os
import platform
from tkinter import Label, Tk, Toplevel
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

from PIL import Image, ImageTk, ImageDraw

if platform.system() == 'Windows':
    import d3dshot
    d = d3dshot.create(capture_output="numpy")

    def grab_screen(x1, y1, x2, y2):
        return d.screenshot(region=(x1, y1, x2, y2))
else:
    LibName = 'prtscn.so'
    AbsLibPath = os.path.dirname(
        os.path.abspath(__file__)) + os.path.sep + LibName
    grab = ctypes.CDLL(AbsLibPath)

    def grab_screen(x1, y1, x2, y2):
        w, h = x2-x1, y2-y1
        size = w * h
        objlength = size * 3

        grab.getScreen.argtypes = []
        result = (ctypes.c_ubyte*objlength)()

        grab.getScreen(x1, y1, w, h, result)
        np_img = np.frombuffer(result, np.uint8).reshape(h, w, 3)
        return np_img


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def thresh_mask(channel):
    return channel[:, :] < 128


def thresh_apply(image):
    if np.average(image) < 128:
        image = 255 - image

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


def clahe_contrast_gray(img):
    clahe = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def scale_contrast(img):
    return cv2.convertScaleAbs(img, alpha=3, beta=0)


def filter_image(image, inv=False):
    contrast = clahe_contrast(image)

    gray = get_grayscale(contrast)
    if inv:
        gray = 255 - gray
    thresh = thresh_apply(gray)

    thresh2 = thresholding(thresh)

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
        text_b = pytesseract.image_to_string(
            b, lang="jpn", timeout=0.5, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 7")
        text_g = pytesseract.image_to_string(
            g, lang="jpn", timeout=0.5, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 7")
        text_r = pytesseract.image_to_string(
            r, lang="jpn", timeout=0.5, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 7")
        return b, g, r, filter_text(text_b), filter_text(text_g), filter_text(text_r)
    except RuntimeError as e:
        print(e)
        return b, g, r, '', '', ''


def text_from_image(image, inv=False):
    img = filter_image(image, inv)
    text = pytesseract.image_to_string(
        img, lang="jpn", timeout=0.5, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 7")
    return img, filter_text(text)


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
            return ("", "", "")
    entry = dictionary_map[text][0]
    return (entry[0], "\n".join(entry[5]), entry[1])


def cursor_search(img):
    b, g, r, text_b, text_g, text_r = text_from_channels(img)

    text_b_s = lookup_text_sudachi(text_b)
    text_g_s = lookup_text_sudachi(text_g)
    text_r_s = lookup_text_sudachi(text_r)
    text_b = lookup_text(text_b)
    text_g = lookup_text(text_g)
    text_r = lookup_text(text_r)

    if len(text_b_s[0]) >= len(text_b[0]):
        text_b = text_b_s
    if len(text_g_s[0]) >= len(text_g[0]):
        text_g = text_g_s
    if len(text_r_s[0]) >= len(text_r[0]):
        text_r = text_r_s

    if text_b[0] != "" and len(text_b[0]) >= len(text_g[0]) and len(text_b[0]) >= len(text_r[0]):
        return text_b
    elif text_g[0] != "" and len(text_g[0]) >= len(text_b[0]) and len(text_g[0]) >= len(text_r[0]):
        return text_g
    elif text_r[0] != "" and len(text_r[0]) >= len(text_b[0]) and len(text_r[0]) >= len(text_g[0]):
        return text_r
    else:
        return ("none", "none", "none")


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

    ctrl_down = False
    mouse_down = False

    select_window = None

    select_label = None
    tooltip_label = None

    tooltip_text = "No text found"

    def __init__(self, root):
        self.root = root
        self.dictionary_map = load_dictionary(
            str(Path(SCRIPT_DIR, 'dictionaries', 'jmdict_english.zip')))

        self.bg_pil = Image.new(
            'RGB', (get_monitors()[0].width, get_monitors()[0].height))
        self.bg_rect = Image.new(
            'RGB', (get_monitors()[0].width, get_monitors()[0].height))
        self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)

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
        if self.ctrl_down:
            jp, en, pron = cursor_search(np.array(self.bg_pil.crop(
                (self.x_min, self.y_min, self.x_max, self.y_max))))
            if jp != "none":
                self.tooltip_text = jp + "\n" + pron + "\n" + en
            else:
                self.tooltip_text = "No text found"

    def mouse_motion(self, event):
        self.mx, self.my = event.x, event.y

    def mouse_motion_button(self, event):
        if self.ctrl_down:
            self.x1, self.y1 = event.x, event.y
            self.mx, self.my = event.x, event.y

            self.order_mouse()

            self.bg_rect = self.bg_pil.copy()
            img_draw = ImageDraw.Draw(self.bg_rect)
            img_draw.rectangle(
                [self.x_min, self.y_min, self.x_max, self.y_max], outline='red')

    def mouse_button(self, event):
        if self.ctrl_down:
            self.mouse_down = True
            self.x, self.y = event.x, event.y
            self.x1 = self.x + 1
            self.y1 = self.y + 1

            self.order_mouse()

    def main(self):
        while True:
            if keyboard.is_pressed("control"):
                if not self.ctrl_down:
                    self.ctrl_down = True

                    bg = grab_screen(
                        0, 0, get_monitors()[0].width, get_monitors()[0].height)
                    self.bg_pil = Image.fromarray(bg)
                    self.bg_rect = self.bg_pil.copy()
                    self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)

                    self.select_window = Toplevel()
                    self.select_window.wm_overrideredirect(True)
                    self.select_window.wm_geometry("%dx%d+%d+%d" %
                                                   (get_monitors()[0].width, get_monitors()[0].height, 0, 0))

                    grab_screen(self.x_min, self.y_min, self.x_max, self.y_max)

                    self.select_label = Label(self.select_window)
                    self.bg_tk = ImageTk.PhotoImage(image=self.bg_rect)
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
                self.bg_tk = ImageTk.PhotoImage(image=self.bg_rect)
                self.select_label.config(image=self.bg_tk)

                self.tooltip_label.place(x=self.mx, y=self.my -
                                         self.tooltip_label.winfo_height() - 10)
                self.tooltip_label.config(text=self.tooltip_text)

                self.select_window.update()
            else:
                self.ctrl_down = False
                if self.select_window is not None:
                    self.select_window.destroy()
                    self.select_window = None
            self.root.update()


SCRIPT_DIR = Path(__file__).parent

win = Tk()
win.overrideredirect(True)
win.geometry("0x0")
win.wait_visibility(win)
win.wm_attributes('-alpha', 0)
ocr = OCR(win)

jam = Jamdict()
dictionary_map = {}
tokenizer_obj = dictionary.Dictionary(dict='full').create()
mode = tokenizer.Tokenizer.SplitMode.A

ocr.main()
