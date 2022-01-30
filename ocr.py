import _thread
import ctypes
import os
from tkinter import Label, Tk, Toplevel
from PIL import Image, ImageTk
import mouse
import keyboard
from screeninfo import get_monitors
import pytesseract
import cv2
import numpy as np
from jamdict import Jamdict
from sudachipy import tokenizer
from sudachipy import dictionary
from pathlib import Path
import zipfile
import json
import platform


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
            # Using headword as key for finding the dictionary entry
            output_map[entry[0]] = [entry]
    return output_map


def setup():
    global dictionary_map
    dictionary_map = load_dictionary(
        str(Path(SCRIPT_DIR, 'dictionaries', 'jmdict_english.zip')))


def grab_screen_linux(x1, y1, x2, y2):
    w, h = x2-x1, y2-y1
    size = w * h
    objlength = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte*objlength)()

    grab.getScreen(x1, y1, w, h, result)
    np_img = np.frombuffer(result, np.uint8).reshape(h, w, 3)
    return np_img


def grab_screen_windows(x1, y1, x2, y2):
    return d.screenshot(region=(x1, y1, x2, y2))

# get grayscale image


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
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
    w = image.shape[1]
    h = image.shape[0]

    b, g, r = cv2.split(image)

    b = thresh_apply(clahe_contrast_gray(b))
    g = thresh_apply(clahe_contrast_gray(g))
    r = thresh_apply(clahe_contrast_gray(r))

    if use_scale_size:
        b = cv2.resize(b, (0, 0), fx=scale_size/min(w, h),
                       fy=scale_size/min(w, h), interpolation=cv2.INTER_NEAREST)
        g = cv2.resize(g, (0, 0), fx=scale_size/min(w, h),
                       fy=scale_size/min(w, h), interpolation=cv2.INTER_NEAREST)
        r = cv2.resize(r, (0, 0), fx=scale_size/min(w, h),
                       fy=scale_size/min(w, h), interpolation=cv2.INTER_NEAREST)

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
    print("looking up: " + text)
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
    print("looking up: " + text)
    if text not in dictionary_map:
        m = tokenizer_obj.tokenize(text, mode)[0]
        text = m.dictionary_form()
        if text not in dictionary_map:
            return ("", "", "")
    entry = dictionary_map[text][0]
    return (entry[0], "\n".join(entry[5]), entry[1])


def cursor_search(x, y, w, h):
    global pic
    if platform.system() == 'Windows':
        img = grab_screen_windows(int(x), int(y), int(x + w), int(y + h))
    else:
        img = grab_screen_linux(int(x), int(y), int(x + w), int(y + h))

    pic = img

    # pic, text = text_from_image(img)

    # text = lookup_text_sudachi(text)

    # return text

    b, g, r, text_b, text_g, text_r = text_from_channels(img)

    # pic = cv2.merge((b, g, r))

    # pic = np.append(np.append(b, g, axis=0), r, axis=0)

    print("raw:\n" + text_b + "\n" + text_g + "\n" + text_r)

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


def mouse_event(event):
    global x, y, x1, y1, x_min, x_max, y_min, y_max
    if isinstance(event, mouse.ButtonEvent) and keyboard.is_pressed("control"):
        if event.event_type == 'down':
            x, y = mouse.get_position()
        if event.event_type == 'up':
            x1, y1 = mouse.get_position()

        x_min = min(x, x1)
        x_max = max(x, x1)
        y_min = min(y, y1)
        y_max = max(y, y1)

        if x_min == x_max:
            x_max += 1
        if y_min == y_max:
            y_max += 1

        if x_max - x_min > max_size:
            if event.event_type == "down":
                x1 = x_min + max_size
            if event.event_type == "up":
                x = x_max - max_size
            x_min = min(x, x1)
            x_max = max(x, x1)
            y_min = min(y, y1)
            y_max = max(y, y1)
        if y_max - y_min > max_size:
            if event.event_type == "down":
                y1 = y_min + max_size
            if event.event_type == "up":
                y = y_max - max_size
            x_min = min(x, x1)
            x_max = max(x, x1)
            y_min = min(y, y1)
            y_max = max(y, y1)


def tesseract_loop():
    global x_min, x_max, y_min, y_max, tooltip_text
    while True:
        if keyboard.is_pressed("control"):
            jp, en, pron = cursor_search(
                x_min, y_min, int(x_max - x_min), int(y_max - y_min))

            print(jp, en, pron)

            if jp != "none":
                print("chose: " + jp)

                tooltip_text = jp + "\n" + pron + "\n" + en


def main_loop():
    global ctrl_x, ctrl_y, label1, tooltip_window, select_window
    mouse.hook(mouse_event)

    while True:
        if keyboard.is_pressed("control"):
            if pressed == False:
                ctrl_x, ctrl_y = mouse.get_position()
                # creates a toplevel window
                tooltip_window = Toplevel()
                # Leaves only the label and removes the app window
                tooltip_window.wm_overrideredirect(True)
                tooltip_window.wm_geometry("+%d+%d" % mouse.get_position())

                label = Label(tooltip_window, text="none", justify='left',
                              background="#ffffff", relief='solid', borderwidth=1,
                              wraplength=180)
                label.pack(ipadx=1)

                # creates a toplevel window
                select_window = Toplevel()
                select_window.wm_overrideredirect(True)
                select_window.wm_geometry("%dx%d+%d+%d" % (get_monitors()
                                                           [0].width, get_monitors()[0].height, 0, 0))
                select_window.wait_visibility(select_window)
                select_window.wm_attributes('-alpha', 0)

            w, h = int(x_max - x_min), int(y_max - y_min)
            win.geometry('%dx%d+%d+%d' %
                         (preview_scale*w, preview_scale*h, ctrl_x - preview_scale*w / 2, ctrl_y - preview_scale*h))

            pilimg = Image.fromarray(pic)
            pilimg = pilimg.resize(
                (preview_scale*w, preview_scale*h), resample=Image.BICUBIC)

            tkimg = ImageTk.PhotoImage(image=pilimg)

            label1.image = tkimg
            label1.configure(image=tkimg)

            label1.place(x=0, y=0, width=preview_scale *
                         w, height=preview_scale*h)

            if tooltip_window:
                label.config(text=tooltip_text)
        else:
            if tooltip_window:
                tooltip_window.destroy()
                tooltip_window = None
            if select_window:
                select_window.destroy()
                select_window = None

            win.geometry('%dx%d+%d+%d' % (0, 0, 0, 0))

        pressed = keyboard.is_pressed("control")

        win.update()


def main():
    _thread.start_new_thread(tesseract_loop, ())

    main_loop()


if platform.system() == 'Windows':
    import d3dshot
    d = d3dshot.create(capture_output="numpy")
else:
    LibName = 'prtscn.so'
    AbsLibPath = os.path.dirname(
        os.path.abspath(__file__)) + os.path.sep + LibName
    grab = ctypes.CDLL(AbsLibPath)

jam = Jamdict()

win = Tk()
win.overrideredirect(True)
win.geometry("0x0")

tokenizer_obj = dictionary.Dictionary(dict='full').create()
mode = tokenizer.Tokenizer.SplitMode.A

SCRIPT_DIR = Path(__file__).parent
dictionary_map = {}

x = y = 0
x1 = y1 = 1
x_min = y_min = 0
x_max = y_max = 1

ctrl_x = ctrl_y = 0

min_size = 10
max_size = 400
scale_size = 128
use_scale_size = False
preview_scale = 2

adjust_speed = 15

label1 = Label(win)
label1.place(x=0, y=0, width=2*(x_max - x_min), height=2*(y_max - y_min))

pressed = False

tooltip_window = None
select_window = None
label = None
pic = np.zeros((1, 1, 3), dtype=np.uint8)

tooltip_text = ""

setup()
main()
