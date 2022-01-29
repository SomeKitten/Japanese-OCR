#include <stdio.h>
#include <X11/X.h>
#include <X11/Xlib.h>
// Compile hint: gcc -shared -O3 -fPIC -o prtscn.so -Wl,-soname,prtscn prtscn_linux.c -lX11

void getScreen(const int, const int, const int, const int, unsigned char *);
void getScreen(const int xx, const int yy, const int W, const int H, /*out*/ unsigned char *data)
{
   Display *display = XOpenDisplay(NULL);
   Window root = DefaultRootWindow(display);

   XImage *image = XGetImage(display, root, xx, yy, W, H, AllPlanes, ZPixmap);

   unsigned long red_mask = image->red_mask;
   unsigned long green_mask = image->green_mask;
   unsigned long blue_mask = image->blue_mask;
   int x, y;
   int ii = 0;
   for (y = 0; y < H; y++)
   {
      for (x = 0; x < W; x++)
      {
         unsigned long pixel = XGetPixel(image, x, y);
         unsigned char blue = (pixel & blue_mask);
         unsigned char green = (pixel & green_mask) >> 8;
         unsigned char red = (pixel & red_mask) >> 16;

         // data[ii] = (blue + green + red) / 3;
         // ii++;

         data[ii] = red;
         data[ii + 1] = green;
         data[ii + 2] = blue;
         ii += 3;
      }
   }
   XDestroyImage(image);
   XDestroyWindow(display, root);
   XCloseDisplay(display);
}
