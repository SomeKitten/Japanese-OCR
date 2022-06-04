import io
import json
from math import floor
import os
import platform
from tkinter import END, Frame, Label, Menu, Text, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import zipfile
from pathlib import Path
from PyPDF2 import PdfFileReader

import cv2
from jamdict import Jamdict
import numpy as np

import pytesseract
from screeninfo import get_monitors

from sudachipy import tokenizer
from sudachipy import dictionary

from PIL import Image, ImageTk, ImageGrab, ImageChops

from img_utils import ImageText

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


def thresh_apply2(image):
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
        return np.array(bg), True, bbox
    else:
        return np.array(im), False, [0, 0, im.size[0], im.size[1]]


def crop(img, bbox, border):
    im = np.full((bbox[3] - bbox[1] + border*2, bbox[2] -
                 bbox[0] + border*2), 255, np.uint8)
    im[border:border+bbox[3]-bbox[1], border:border+bbox[2] -
        bbox[0]] = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    return im


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

    return purged_cnts, bg


def line_contour_ext(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    bg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    purged_cnts = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > (img.shape[0] / 20 * img.shape[1] / 20) and area < (img.shape[0] / 3 * img.shape[1] / 3):
            purged_cnts.append(cnt)

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
    bounds = []

    start = -1
    last = 0
    for c, col in enumerate(img.T):
        if np.min(col) < 255:
            continue

        start = start + 1
        if c - start > len(img.T) / 10:
            imgs.append(img[:, start:c])
            bounds.append((start, c))
            last = c
        start = c

    if np.average(img.T[-1]) != 255 and len(img.T) - last > len(img.T) / 10:
        if last == 0:
            imgs.append(img)
            bounds.append((0, len(img.T)))
        else:
            imgs.append(img[:, start:])
            bounds.append((start, len(img.T)))

    if len(imgs) == 0:
        return [], []

    max_len = [len(i.T) for i in imgs]
    max_len = np.max(max_len)

    purged_imgs = []
    purged_bounds = []

    for i, im in enumerate(imgs):
        if len(im.T) > max_len * 0.8:
            purged_imgs.append(im)
            purged_bounds.append(bounds[i])

    purged_imgs = purged_imgs[::-1]
    purged_bounds = purged_bounds[::-1]

    return purged_imgs, purged_bounds


def filter_image(image):
    scl = scale(image)

    cont = idk_contrast(scl)

    thresh = thresh_apply2(cont)

    removed = remove_edge_lines(thresh)

    trimmed, success, bbox = trim(removed, 0)

    origcropped = crop(scl, bbox, 0) if success else scl.copy()

    images, bounds = furigana_removal(trimmed)

    for i, img in enumerate(images):
        bound = bounds[i]
        origt = origcropped[:, bound[0]:bound[1]]

        t, s, b = trim(img, 10)
        origc = crop(origt, b, 10) if s else origt.copy()

        and_result = cv2.bitwise_not(cv2.bitwise_and(
            cv2.bitwise_not(t), cv2.bitwise_not(origc)))

        images[i] = and_result

    return images


def get_text_bubbles(image):
    print("Getting text bubbles!")

    gray = get_grayscale(image)

    # preprocessing
    thresh = thresh_apply(gray)
    inv = cv2.bitwise_not(thresh)
    dil = dilate(inv)
    inv2 = cv2.bitwise_not(dil)

    cv2.imwrite("tmp/bubbles.png", inv2)

    # bubble detection
    cont = line_contour_tree(inv2)

    cv2.imwrite("tmp/bubbles_cont.png", cont)

    cnts, cont2 = line_contour_tree2(cont)

    cv2.imwrite("tmp/bubbles_cont2.png", cont2)

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

        filtered = filter_image(cropped_gray)

        for f, filt in enumerate(filtered):
            cv2.imwrite("tmp/bubble_{}_{}.png".format(c, f), filt)

        text = text_from_lines(filtered, c)

        if len(text) < 2:
            continue

        # bubbles.append((bound, text))
        bubbles.append((bound, ''))

    print("Found {} bubbles!".format(len(bubbles)))

    return bubbles


def text_from_lines(imgs, c):
    full_text = ""

    for i, img in enumerate(imgs):
        data = pytesseract.image_to_data(
            img, lang="jpn_vert", timeout=2, config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_()0123456789 --oem 1 --psm 5", output_type=pytesseract.Output.DICT)

        text = filter_data(data)

        if len(text) == 0:
            continue

        full_text += text + "\n"
        # cv2.imwrite("tmp/{}-{}.png".format(c, i), img)

    return full_text.strip()


def filter_text(text):
    # TODO replace with blacklist of characters
    text = text.replace('\n', '')
    text = text.replace(' ', '')
    text = text.replace('\\', '')
    text = text.replace('/', '')
    text = text.replace('+', '')
    text = text.replace('-', '')
    text = text.replace('=', '')
    text = text.replace('<', '')
    text = text.replace('>', '')

    text = text.replace('|', 'ー')

    text = text.strip()
    return text


def filter_data(data):
    text = ""

    conf = data["conf"][::-1]
    start = len(data["conf"]) - (conf.index('-1')
                                 if '-1' in conf else conf.index(-1))
    for i in range(start, len(data["conf"])):
        # to remove duplicate kana
        if data["conf"][i] < 30:
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
    closed = False

    images = []

    bubbles = []
    boxes = []
    jp_text = []
    en_text = []

    show_bubbles = True
    show_original = True
    auto_ocr = False
    been_ocr_pages = []

    scale = 1

    page = 0
    file = False

    def __init__(self, root):
        print("Loading dictionary...")
        self.dictionary_map = load_dictionary(
            str(Path(SCRIPT_DIR, 'dictionaries', 'jmdict_english.zip')))

        self.root: TkinterDnD.Tk = root

        self.root.title("Manga Translation")
        self.root.wm_geometry("%dx%d+%d+%d" %
                              (get_monitors()[0].width, get_monitors()[0].height, 0, 0))

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        print("Initializing defaults for image processing...")

        self.bg = Label(self.root, borderwidth=0)
        self.bg.place(x=0, y=0)

        self.bg_pil = Image.new(
            'RGB', (self.root.winfo_width(), self.root.winfo_height()), (0, 0, 0))
        self.images = [Image.new(
            'RGB', (self.root.winfo_width(), self.root.winfo_height())), (0, 0, 0)]

        self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)

        self.bg.config(image=self.bg_tk)

        self.menu = Menu(self.root)

        self.file_menu = Menu(self.menu, tearoff=0)
        # self.file_menu.add_command(label="Open", command=self.open_file)
        # self.file_menu.add_command(label="Save", command=self.save_file)
        # self.file_menu.add_command(label="Save as", command=self.save_file_as)
        self.file_menu.add_command(label="Import", command=self.import_pdf)
        self.file_menu.add_command(label="Export", command=self.export_pdf)
        self.file_menu.add_command(label="Exit", command=self.on_close)
        self.menu.add_cascade(label="File", menu=self.file_menu)

        self.edit_menu = Menu(self.menu, tearoff=0)
        self.edit_menu.add_command(
            label="Toggle bubbles", command=self.toggle_bubbles)
        # self.edit_menu.add_command(
        #     label="Toggle JP/EN edit", command=self.toggle_lang)
        self.menu.add_cascade(label="Edit", menu=self.edit_menu)

        # self.root.bind("<Tab>", self.tab_key)

        # Windows + OSX
        self.root.bind("<Shift-Tab>", self.shift_tab_key)
        # Linux
        self.root.bind("<Shift-ISO_Left_Tab>", self.shift_tab_key)

        self.root.config(menu=self.menu)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # def open_file(self):
    #     pass

    # def save_file(self):
    #     pass

    # def save_file_as(self):
    #     pass

    # TODO export PDF instead of (temporary) PNG
    def export_pdf(self):
        img = self.images[0]
        img_txt = ImageText(img)

        for b, box in enumerate(self.boxes):
            if self.en_text[b] == "":
                continue

            img_txt.draw.rectangle(
                (box[0], box[1], box[0]+box[2], box[1]+box[3]), fill=(255, 255, 255))
            img_txt.write_text_box(
                (box[0], box[1]), self.en_text[b], box[2] * 0.9, 'Roboto-Bold.ttf', font_size=int(box[2] * 0.2), color=(0, 0, 0), place='center')

            print(self.en_text[b])

        img_txt.save("tmp/output.png")

    def import_pdf(self):
        path = filedialog.askopenfilename(
            title="Select PDF file", filetypes=[("PDF", "*.pdf")])
        if path == '':
            return
        self.load_pdf(path)

    def tab_key(self, event):
        self.toggle_lang()
        return 'break'

    def shift_tab_key(self, event):
        self.toggle_bubbles()
        return 'break'

    def toggle_bubbles(self):
        self.show_bubbles = not self.show_bubbles
        if self.show_bubbles:
            for bubble in self.bubbles:
                bubble.lift()
        else:
            self.bg.lift()

    def toggle_lang(self):
        if self.been_ocr_pages[self.page]:
            self.show_original = not self.show_original

            if self.show_original:
                for i, bubble in enumerate(self.bubbles):
                    text = bubble.winfo_children()[0]
                    self.en_text[i] = text.get('1.0', 'end-1c')
                    text.delete('1.0', 'end')
                    text.insert('end', self.jp_text[i])
            else:
                for i, bubble in enumerate(self.bubbles):
                    text = bubble.winfo_children()[0]
                    self.jp_text[i] = text.get('1.0', 'end-1c')
                    text.delete('1.0', 'end')
                    text.insert('end', self.en_text[i])

    def update_text(self, event):
        if self.show_original:
            for i, bubble in enumerate(self.bubbles):
                text = bubble.winfo_children()[0]
                self.jp_text[i] = text.get('1.0', 'end-1c')
        else:
            for i, bubble in enumerate(self.bubbles):
                text = bubble.winfo_children()[0]
                self.en_text[i] = text.get('1.0', 'end-1c')

    def on_resize(self):
        button_width = self.hotkey_button.winfo_width()
        button_height = self.hotkey_button.winfo_height()

        self.button_font['size'] = min(button_width // 16, button_height // 2)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    def convert_page(self, file, p):
        # TODO cache for converted images
        print("Starting converting page!")
        self.reader = PdfFileReader(file)
        page = self.reader.pages[p]
        xObject = page['/Resources']['/XObject'].getObject()

        print("A")

        for obj in xObject:
            print(obj)
            if xObject[obj]['/Subtype'] == '/Image':
                print("B")
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    print("C")
                    mode = "RGB"
                else:
                    print("D")
                    mode = "P"

                print(xObject[obj]['/Filter'])

                print("E")

                print(data)

                image = Image.open(io.BytesIO(data))

                print(image)

                while len(self.images) < p + 1:
                    self.images.append(False)
                self.images[p] = image

                self.bg_pil = Image.new(
                    'RGB', (self.bg.winfo_width(), self.bg.winfo_height()))

                self.scale = self.bg.winfo_height() / size[1]

                print("Done converting page!")

                for bubble in self.bubbles:
                    bubble.destroy()

                bbls = get_text_bubbles(np.array(self.images[p]))

                self.bubbles, self.boxes, self.jp_text, self.en_text = self.create_bubbles(
                    bbls)

                self.toggle_lang()

                self.bg_pil = self.images[p].resize(
                    (floor(self.images[p].width * self.scale), self.bg.winfo_height()))

                self.bg_tk = ImageTk.PhotoImage(image=self.bg_pil)
                self.bg.config(image=self.bg_tk)

                return
        raise Exception("No image found!")

    def create_bubbles(self, bbls):
        self.show_bubbles = True

        bubbles = []
        boxes = []
        jp = []
        en = []
        for box, text in bbls:
            bubbles.append(self.create_bubble(box, ''))
            boxes.append(box)
            jp.append(text)
            en.append("")

        return bubbles, boxes, jp, en

    def create_bubble(self, bubble, text):
        text_frame = Frame(self.root, width=int(
            bubble[2]*self.scale), height=int(bubble[3]*self.scale))
        text_frame.place(x=int(bubble[0]*self.scale),
                         y=int(bubble[1]*self.scale))

        text_box = Text(text_frame, bg='white')
        text_box.insert(END, text)
        text_box.place(x=0, y=0, width=int(
            bubble[2]*self.scale), height=int(bubble[3]*self.scale))

        text_box.bind("<Button-1>", lambda _: text_frame.lift())
        text_box.bind("<Tab>", self.tab_key)
        text_box.bind("<Key>", self.update_text)

        # Windows + OSX
        text_box.bind("<Shift-Tab>", self.shift_tab_key)
        # Linux
        text_box.bind("<Shift-ISO_Left_Tab>", self.shift_tab_key)

        text_box.bind("<Control-Right>",
                      lambda _: self.switch_page(self.page + 1))

        return text_frame

    def on_drop(self, event: TkinterDnD.DnDEvent):
        file = event.data.split(
            "} {")[0].replace("{", "").replace("}", "")
        self.load_pdf(file)

    def load_pdf(self, file):
        self.page = 0
        self.file = file

        self.show_original = self.auto_ocr
        self.been_ocr_pages = [False for _ in range(self.page + 1)]
        self.convert_page(file, self.page)
        self.been_ocr_pages[self.page] = self.auto_ocr

    def switch_page(self, page):
        self.show_original = self.auto_ocr
        self.page = page

        while len(self.been_ocr_pages) < page + 1:
            self.been_ocr_pages.append(False)

        self.convert_page(self.file, page)
        self.been_ocr_pages[page] = self.auto_ocr

    def main(self):
        print("Initialization complete!")
        while True:
            if self.closed:
                return

            self.root.update()


tmpdir = 'tmp'
for f in os.listdir(tmpdir):
    os.remove(os.path.join(tmpdir, f))


print("Loading jamdict...")
jam = Jamdict()
dictionary_map = {}

print("Loading sudachidict_full...")
tokenizer_obj = dictionary.Dictionary(dict="full").create()
mode = tokenizer.Tokenizer.SplitMode.B


print("Starting...")
ocr = OCR(TkinterDnD.Tk())

ocr.main()
