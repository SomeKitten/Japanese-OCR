import asyncio
from curses.textpad import Textbox
import enum
import json
from math import floor
import os
import platform
from tkinter import BOTH, TOP, Button, Label, Tk, Toplevel, font
import tkinter
from tkinterdnd2 import DND_FILES, TkinterDnD
import zipfile
from pathlib import Path
from pdf2image import convert_from_path
import tempfile

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
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def dilate2(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=3)


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


def thresh_apply2(image):
    thresh = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    mask = image[:, :] > 100

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
    arr = img[img != 0]

    min = np.min(arr)
    max = np.max(arr)

    cont = 255 / (max - min) * (img - min)

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
    if min(img.shape) < 200:
        scl = 200 / min(img.shape)
        return cv2.resize(img, (int(img.shape[1] * scl), int(img.shape[0] * scl)), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def scale2(img):
    if min(img.shape) > 100:
        return cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def trim(im, border):
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


def line_contour_tree(img):
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    bg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    purged_cnts = []

    for c, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > (img.shape[0] / 20 * img.shape[1] / 20) and area < (img.shape[0] / 3 * img.shape[1] / 3):
            purged_cnts.append(cnt)

    cv2.drawContours(bg, purged_cnts, -1, 255, 3)

    return bg


def line_contour_tree2(img):
    cnts, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    purged_cnts = []

    for c, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > (img.shape[0] / 20 * img.shape[1] / 20) and area < (img.shape[0] / 3 * img.shape[1] / 3) and hierarchy[0][c][2] == -1:
            purged_cnts.append(cnt)

    cv2.drawContours(bg, purged_cnts, -1, 255, 3)

    return purged_cnts


def line_contour_ext(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    bg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    purged_cnts = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > (img.shape[0] / 20 * img.shape[1] / 20) and area < (img.shape[0] / 3 * img.shape[1] / 3):
            purged_cnts.append(cnt)

    print(cnts)
    print(purged_cnts)

    cv2.drawContours(bg, cnts, -1, 255, 3)

    return bg


def sharpen(image):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def unsharp_mask(image):
    amount = 5.0
    blurred = cv2.GaussianBlur(image, (3, 3), 5.0)
    return (amount + 1) * image - amount * blurred


def remove_edge_lines(img):
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for cnt in cnts:
        rect = cv2.boundingRect(cnt)
        if rect[0] < img.shape[1] / 2 and rect[0] + rect[2] > img.shape[1] / 2 and rect[1] < img.shape[0] / 2 and rect[1] + rect[3] > img.shape[0] / 2:
            cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)

    out_img = 255-cv2.bitwise_and(255-img, mask)

    return out_img


def furigana_removal(img):
    imgs = []

    start = -1
    last = 0
    for c, col in enumerate(img.T):
        if np.min(col) < 255:
            continue

        start = start + 1
        if c - start > len(img.T) / 10:
            imgs.append(img.T[start:c].T)
            last = c
        start = c

    if np.average(img.T[-1]) != 255 and len(img.T) - last > len(img.T) / 10:
        if last == 0:
            imgs.append(img)
        else:
            imgs.append(img.T[start:].T)

    if len(imgs) == 0:
        return imgs

    max_len = [len(i.T) for i in imgs]
    max_len = np.max(max_len)

    purged_imgs = []

    for i in imgs:
        if len(i.T) > max_len * 0.8:
            purged_imgs.append(i)

    purged_imgs = purged_imgs[::-1]

    return purged_imgs


def filter_image(image):
    scl = scale(image)

    cont = idk_contrast(scl)

    thresh = thresh_apply2(cont)

    removed = remove_edge_lines(thresh)

    trimmed = trim(removed, 0)

    images = furigana_removal(trimmed)

    for i, img in enumerate(images):
        inv = 255 - img
        thin = cv2.ximgproc.thinning(inv)
        dil = dilate(thin)
        inv2 = 255 - dil
        images[i] = trim(inv2, 30)

    return images


def get_text_bubbles(image):
    print("Getting text bubbles!")

    gray = get_grayscale(image)

    # preprocessing
    thresh = thresh_apply(gray)
    inv = cv2.bitwise_not(thresh)
    dil = dilate(inv)
    inv2 = cv2.bitwise_not(dil)

    # bubble detection
    cont = line_contour_tree(inv2)
    cnts = line_contour_tree2(cont)

    # postprocessing(?)

    bubbles = []

    for c, cnt in enumerate(cnts):
        crop1 = 0.1
        crop2 = 0.15

        bound = cv2.boundingRect(cnt)

        cropped_gray = gray.copy()[bound[1]:bound[1] + bound[3],
                                   bound[0]:bound[0] + bound[2]]

        cropped = gray.copy()[bound[1]:bound[1] + bound[3],
                              bound[0]:bound[0] + bound[2]]
        cropped1 = gray.copy()[int(bound[1] + bound[3]*crop1):int(bound[1] + bound[3]*(1-crop1)),
                               int(bound[0] + bound[2]*crop1):int(bound[0] + bound[2]*(1-crop1))]
        cropped2 = gray.copy()[int(bound[1] + bound[3]*crop2):int(bound[1] + bound[3]*(1-crop2)),
                               int(bound[0] + bound[2]*crop2):int(bound[0] + bound[2]*(1-crop2))]

        cropped[cropped < 250] = 0

        cropped1[cropped1 < 250] = 0
        cropped2[cropped2 < 250] = 0

        avg = np.average(cropped)

        avg1 = np.average(cropped1)
        avg2 = np.average(cropped2)

        if not (avg > 128 and avg1 > avg2):
            continue

        filtered = filter_image(cropped_gray)
        text = text_from_lines(filtered, c)

        if text == "":
            continue

        bubbles.append((bound, text))

        # print(c, text)

        # print(bound)
        # cv2.rectangle(image, (bound[0], bound[1]),
        #               (bound[0] + bound[2], bound[1] + bound[3]), (0, 255, 0), 2)
        # cv2.rectangle(image, (int(bound[0] + bound[2]*crop1), int(bound[1] + bound[3]*crop1)),
        #               (int(bound[0] + bound[2]*(1-crop1)), int(bound[1] + bound[3]*(1-crop1))), (255, 0, 0), 2)
        # cv2.rectangle(image, (int(bound[0] + bound[2]*crop2), int(bound[1] + bound[3]*crop2)),
        #               (int(bound[0] + bound[2]*(1-crop2)), int(bound[1] + bound[3]*(1-crop2))), (255, 0, 0), 2)

    print("Found {} bubbles!".format(len(bubbles)))

    return bubbles


def text_from_lines(imgs, c):
    full_text = ""

    for i, img in enumerate(imgs):
        data = pytesseract.image_to_data(
            img, lang="jpn_vert", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ --oem 1 --psm 5", output_type=pytesseract.Output.DICT)

        text = filter_data(data)

        if len(text) == 0:
            continue

        full_text += text + "\n"
        cv2.imwrite("tmp/{}-{}.png".format(c, i), img)

    return full_text.strip()


def filter_text(text):
    text = text.replace('\n', '')
    text = text.replace(' ', '')
    text = text.replace('|', 'ー')
    text = text.replace(')', 'ー')
    text = text.strip()
    return text


def filter_data(data):
    text = ""
    conf = data["conf"][::-1]
    start = len(data["conf"]) - (conf.index('-1')
                                 if '-1' in conf else conf.index(-1))
    for i in range(start, len(data["conf"])):
        # to remove duplicate kana
        if data["conf"][i] < np.mean(data["conf"][start:]) * 0.5:
            if len(data["text"][i]) > 1:
                text += data["text"][i][0]
            continue

        text += data["text"][i]

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


def definition(text):
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

    # key_down = False
    # mouse_down = False

    closed = False

    # hotkey = ""

    bubbles = []
    scale = 1

    def __init__(self, root):
        self.root: TkinterDnD.Tk = root

        self.root.title("Manga Translation")
        self.root.wm_geometry("%dx%d+%d+%d" %
                              (get_monitors()[0].width, get_monitors()[0].height, 0, 0))

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        print("Initializing defaults for image processing...")

        self.bg = Label(self.root)
        self.bg.place(x=0, y=0)

        self.bg_pil = Image.new(
            'RGB', (self.root.winfo_width(), self.root.winfo_height()), (0, 0, 0))
        self.images = [Image.new(
            'RGB', (self.root.winfo_width(), self.root.winfo_height())), (0, 0, 0)]

        self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)

        self.bg.config(image=self.bg_tk)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        print("Loading dictionary...")
        self.dictionary_map = load_dictionary(
            str(Path(SCRIPT_DIR, 'dictionaries', 'jmdict_english.zip')))

    def on_resize(self, event):
        button_width = self.hotkey_button.winfo_width()
        button_height = self.hotkey_button.winfo_height()

        self.button_font['size'] = min(button_width // 16, button_height // 2)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    def convert_page(self, file, page):
        # TODO cache for converted images
        print("Starting converting page!")
        self.images = convert_from_path(file, first_page=page, last_page=page)

        self.bg_pil = Image.new(
            'RGB', (self.bg.winfo_width(), self.bg.winfo_height()))

        self.scale = self.bg.winfo_height() / self.images[0].height

        # self.images[0] = self.images[0].resize(
        #     (floor(self.images[0].width * self.scale), self.bg_pil.height))
        # self.bg_pil.paste(self.images[0], (0, 0))

        print("Done converting page!")

    def create_bubbles(self, bubbles):
        for bubble, text in bubbles:
            print(bubble, text)

            self.create_bubble(bubble, text)

    def create_bubble(self, bubble, text):
        text_frame = tkinter.Frame(self.root, width=int(
            bubble[2]*self.scale), height=int(bubble[3]*self.scale))
        text_frame.place(x=int(bubble[0]*self.scale),
                         y=int(bubble[1]*self.scale))
        # text_frame.grid_propagate(False)

        text_box = tkinter.Text(text_frame, bg='white')
        text_box.insert(tkinter.END, text)
        text_box.place(x=0, y=0, width=int(
            bubble[2]*self.scale), height=int(bubble[3]*self.scale))

        def on_click(event):
            text_frame.lift()

        text_box.bind("<Button-1>", on_click)

    def on_drop(self, event: TkinterDnD.DnDEvent):
        file = event.data.split(
            "} {")[0].replace("{", "").replace("}", "")

        self.convert_page(file, 12)

        self.bubbles = get_text_bubbles(np.array(self.images[0]))

        self.create_bubbles(self.bubbles)

        self.bg_pil = self.images[0].resize(
            (floor(self.images[0].width * self.scale), self.bg.winfo_height()))

        self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)
        self.bg.config(image=self.bg_tk)

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

            self.root.update()


print("Loading jamdict...")
jam = Jamdict()
dictionary_map = {}

print("Loading sudachidict_full...")
tokenizer_obj = dictionary.Dictionary(dict="full").create()
mode = tokenizer.Tokenizer.SplitMode.B


print("Starting...")
ocr = OCR(TkinterDnD.Tk())

ocr.main()
