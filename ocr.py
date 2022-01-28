import ctypes
import os
from time import sleep
from tkinter import Label, Tk, Toplevel
from PIL import Image, ImageTk
import mouse
import keyboard
from screeninfo import get_monitors
import pytesseract
import cv2
import numpy as np
from jamdict import Jamdict

LibName = 'prtscn.so'
AbsLibPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + LibName
grab = ctypes.CDLL(AbsLibPath)

jam = Jamdict()

win = Tk()
win.overrideredirect(True)
win.geometry("0x0")


def grab_screen(x1, y1, x2, y2):
    w, h = x2-x1, y2-y1
    size = w * h
    objlength = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte*objlength)()

    grab.getScreen(x1, y1, w, h, result)
    # return Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)
    np_img = np.frombuffer(result, np.uint8).reshape(h, w, 3)
    return np_img


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
    if np.average(channel) < 128:
        channel = 255 - channel

    return channel[:, :] < 128
    # mask = channel[:, :] < 128
    # thresh = channel
    # thresh[mask] = 0


def thresh_apply(image):
    mask = image[:, :] < 128
    image[mask] = 0
    return image


def clahe_contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def filter_image(image, inv=False):
    # if np.average(gray) < 128:
    #     gray = 255 - gray

    # mask = gray[:, :] < 128
    # thresh = gray
    # thresh[mask] = 0

    contrast = clahe_contrast(image)

    # red = thresh_mask(contrast[:, :, 2])
    # green = thresh_mask(contrast[:, :, 1])
    # blue = thresh_mask(contrast[:, :, 0])

    # thresh = get_grayscale(contrast)
    # thresh[np.logical_and(np.logical_and(red, green), blue)] = 0

    gray = get_grayscale(contrast)
    if inv:
        gray = 255 - gray
    thresh = thresh_apply(gray)

    return thresh


def text_from_image(image, inv=False):
    result = filter_image(image, inv)

    img = result

    text = pytesseract.image_to_string(
        img, lang="jpn", config="-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz --oem 1 --psm 6")

    text = text.replace('\n', '')
    text = text.replace(' ', '')
    text = text.strip()

    return img, text


def cursor_search(x, y, w, h):
    if x < 0.5*w:
        x = 0.5*w
    if y < 0.5*h:
        y = 0.5*h
    if x > get_monitors()[0].width - 0.5*w:
        x = get_monitors()[0].width - 0.5*w
    if y > get_monitors()[0].height - 0.5*h:
        y = get_monitors()[0].height - 0.5*h

    if y > get_monitors()[0].height - 3*h:
        win.geometry('%dx%d+%d+%d' %
                     (2*w, 2*h, x - w, y - 3*h))
    else:
        win.geometry('%dx%d+%d+%d' %
                     (2*w, 2*h, x - w, y + h))

    pic = grab_screen(int(x - 0.5*w), int(y - 0.5*h),
                      int(x + 0.5*w), int(y + 0.5*h))

    img, text = text_from_image(pic)

    pilimg = Image.fromarray(img)
    pilimg = pilimg.resize((2*w, 2*h), resample=Image.NEAREST)

    tkimg = ImageTk.PhotoImage(image=pilimg)

    label1.image = tkimg
    label1.configure(image=tkimg)

    label1.place(x=0, y=0, width=2*w, height=2*h)

    win.update()

    print("raw: " + text)

    if text == '':
        return ("none", "none", "none")
    look = jam.lookup(text)
    while len(look.entries) == 0 or len(look.entries[0].senses) == 0:
        text = text[:-1]

        if text == '':
            return ("none", "none", "none")

        look = jam.lookup(text)
    return (text, look.entries[0].senses[0].text().replace("/", "\n"), look.entries[0].kana_forms[0].text)


class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """

    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # miliseconds
        self.wraplength = 180  # pixels
        # self.widget = widget
        self.text = text
        # self.widget.bind("<Enter>", self.enter)
        # self.widget.bind("<Leave>", self.leave)
        # self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background="#ffffff", relief='solid', borderwidth=1,
                      wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


width = 50
height = 25

min = 10
max = 200

adjust_speed = 15

label1 = Label(win)
label1.place(x=0, y=0, width=2*width, height=2*height)

pressed = False

tw = None
label = None

while True:
    x, y = mouse.get_position()

    # mouse.hook(scroll)

    if keyboard.is_pressed('left'):
        width -= adjust_speed
        if width < min:
            width = min
    if keyboard.is_pressed('right'):
        width += adjust_speed
        if width > max:
            width = max
    if keyboard.is_pressed('down'):
        height -= adjust_speed
        if height < min:
            height = min
    if keyboard.is_pressed('up'):
        height += adjust_speed
        if height > max:
            height = max

    if keyboard.is_pressed("control"):
        if pressed == False:
            # creates a toplevel window
            tw = Toplevel()
            # Leaves only the label and removes the app window
            tw.wm_overrideredirect(True)
            tw.wm_geometry("+%d+%d" % (x, y))
            label = Label(tw, text="none", justify='left',
                          background="#ffffff", relief='solid', borderwidth=1,
                          wraplength=180)
            label.pack(ipadx=1)
        jp, en, pron = cursor_search(x, y, int(width), int(height))

        if jp != "none":
            print(jp, en, pron)

        label.config(text=jp + "\n" + pron + "\n" + en)

        sleep(1/30)
    else:
        if tw:
            tw.destroy()
            tw = None

        win.geometry('%dx%d+%d+%d' % (0, 0, 0, 0))

    pressed = keyboard.is_pressed("control")

    win.update()
