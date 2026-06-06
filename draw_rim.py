from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys, os

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

if len(sys.argv) != 2:
    print("Usage: python draw_rim.py <target_file>")
    sys.exit(1)

target_file = sys.argv[1]

im0 = Image.open(target_file).convert('RGB')
im = np.array(im0)

def rgb_to_y(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

im_y = rgb_to_y(im[:, :, 0], im[:, :, 1], im[:, :, 2])
if os.path.exists(target_file + '.rimline.png'):
    rimline0 = Image.open(target_file + '.rimline.png').convert('L')
    rim_map = np.array(rimline0) > 0
else:
    rim_map = np.zeros(im_y.shape, dtype=bool)
prev_rim_map = rim_map

fig = plt.figure("(Rim line) Space: save, u: undo")
grid = fig.add_gridspec(30, 10, left=0, right=1, top=1, bottom=0)
ax = fig.add_subplot(grid[2:, :])
ax_button = fig.add_subplot(grid[0, 0])
ax_button2 = fig.add_subplot(grid[0, 1])
ax_slider_y = fig.add_subplot(grid[0, 3:7])
ax_slider_dist = fig.add_subplot(grid[1, 3:7])
ax_radio = fig.add_subplot(grid[0:2, 8])
ax_button = Button(ax_button, 'Pen')
ax_button2 = Button(ax_button2, 'Fill')
ax_slider_y = Slider(ax_slider_y, 'Y threshold', 0, 255, valinit=50, valstep=1)
ax_slider_dist = Slider(ax_slider_dist, 'dist', 1, 25, valinit=1, valstep=1)
ax_radio = RadioButtons(ax_radio, ['range','high','low'])
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, sharex=ax, sharey=ax)
ax.imshow(np.where(rim_map[:, :, None], [255, 0, 0], im))
ax2.imshow(rim_map)
pen_mode = True
rect_mode = False
line_points = []
closed = False

def drawimage():
    global line_points, closed
    if not closed:
        return
    im1 = np.zeros(im.shape[:2], dtype=np.uint8)
    im1 = Image.fromarray(im1)
    draw = ImageDraw.Draw(im1)
    xy = []

    for points in line_points:
        if len(points) == 2:
            xy += [points[0][::-1], points[1][::-1]]

    draw.polygon(xy, outline=255, fill=255)
    return np.array(im1)

def on_button_click(event):
    global pen_mode
    pen_mode = not pen_mode
    print("Pen mode:", "PEN" if pen_mode else "ERRASER")
    if pen_mode:
        ax_button.label.set_text('Pen')
    else:
        ax_button.label.set_text('Eraser')
    fig.canvas.draw()

def on_button2_click(event):
    global rect_mode, line_points
    rect_mode = not rect_mode
    print("Rect mode:", "RECT" if rect_mode else "FILL")
    line_points = []
    if rect_mode:
        ax_button2.label.set_text('Rect')
    else:
        ax_button2.label.set_text('Fill')

    ax.clear()
    ax.imshow(np.where(rim_map[:, :, None], [255, 0, 0], im))
    fig.canvas.draw()

def on_press(event):
    global rim_map, prev_rim_map, closed, line_points
    sys.stdout.flush()
    if event.key == ' ':
        print("Saved.")
        Image.fromarray(rim_map).convert('L').save(target_file + '.rimline.png')
        return

    if event.key == 'u':
        rim_map = prev_rim_map
        ax.clear()
        ax2.clear()
        ax.imshow(np.where(rim_map[:, :, None], [255, 0, 0], im))
        ax2.imshow(rim_map)
        fig.canvas.draw()
        fig2.canvas.draw()

    if not closed and len(line_points) > 2 and event.key == 'enter':
        closed = True
        line_points.append([line_points[-1][-1], line_points[0][0]])

        prev_rim_map = rim_map.copy()
        if pen_mode:
            rim_map |= drawimage() > 0
        else:
            rim_map &= ~(drawimage() > 0)

        line_points = []
        closed = False

        ax.clear()
        ax2.clear()
        ax.imshow(np.where(rim_map[:, :, None], [255, 0, 0], im))
        ax2.imshow(rim_map)

        fig.canvas.draw()
        fig2.canvas.draw()

def onclick(event):
    global rect_mode, rim_map, prev_rim_map, line_points, closed
    if not event.dblclick and not rect_mode:
        return
    x = event.xdata
    y = event.ydata
    d = ax_slider_dist.val
    y_threshold = ax_slider_y.val
    if x is not None and y is not None:
        print(f"Clicked at: ({x:.2f}, {y:.2f})")
        xi = int(x)
        yi = int(y)
        if 0 <= xi < im.shape[1] and 0 <= yi < im.shape[0]:
            if rect_mode:
                if event.button == 3:
                    for li, points in enumerate(line_points):
                        for i, (px, py) in enumerate(points):
                            if abs(px - y) < 5 and abs(py - x) < 5:
                                if i == 0:
                                    if li > 0:
                                        line_points[li-1][-1] = points[-1]
                                        line_points.pop(li)
                                    elif closed:
                                        line_points[-1][-1] = points[-1]
                                        line_points.pop(li)
                                    else:
                                        line_points.pop(li)
                                elif i == len(points) - 1:
                                    if li < len(line_points) - 1:
                                        line_points[li+1][0] = points[0]
                                        line_points.pop(li)
                                    elif closed:
                                        line_points[0][0] = points[0]
                                        line_points.pop(li)
                                    else:
                                        line_points.pop(li)
                                else:
                                    points.pop(i)
                                break
                        else:
                            continue
                        break
                    if len(line_points) < 3:
                        closed = False
                elif event.dblclick:
                    if len(line_points) == 0:
                        line_points.append([(y, x)])
                    elif len(line_points[-1]) == 1:
                        line_points[-1].append((y, x))
                    else:
                        line_points.append([line_points[-1][-1], (y, x)])
                    if len(line_points) > 2 and abs(line_points[0][0][0] - line_points[-1][-1][0]) < 5 and abs(line_points[0][0][1] - line_points[-1][-1][1]) < 5:
                        closed = True
                        line_points[-1][-1] = line_points[0][0]
                else:
                    return

                if closed and len(line_points) > 2:
                    prev_rim_map = rim_map.copy()
                    if pen_mode:
                        rim_map |= drawimage() > 0
                    else:
                        rim_map &= ~(drawimage() > 0)

                    line_points = []
                    closed = False
            else:
                base_y = im_y[yi, xi]
                if ax_radio.index_selected == 0:
                    search_map = np.zeros(im_y.shape, dtype=bool)
                    stack = [(yi, xi)]
                    while stack:
                        cy, cx = stack.pop()
                        if search_map[cy, cx]:
                            continue
                        search_map[cy, cx] = True
                        for ny in range(cy - d, cy + d + 1):
                            for nx in range(cx - d, cx + d + 1):
                                if (0 <= ny < im.shape[0] and 0 <= nx < im.shape[1] and
                                    not search_map[ny, nx] and
                                    abs(im_y[ny, nx] - base_y) < y_threshold):
                                    stack.append((ny, nx))
                elif ax_radio.index_selected == 1:
                    search_map = im_y >= base_y
                elif ax_radio.index_selected == 2:
                    search_map = im_y <= base_y
                prev_rim_map = rim_map.copy()
                if pen_mode:
                    rim_map |= search_map
                else:
                    rim_map &= ~search_map

            ax.clear()
            ax2.clear()
            ax.imshow(np.where(rim_map[:, :, None], [255, 0, 0], im))
            ax2.imshow(rim_map)

            for points in line_points:
                if closed:
                    ax.scatter([p[1] for p in points], [p[0] for p in points], c='red')
                else:
                    ax.scatter([p[1] for p in points], [p[0] for p in points], c='green')
            for points in line_points:
                if len(points) > 1:
                    ax.plot([points[0][1], points[1][1]], [points[0][0], points[1][0]], c='blue')

            fig.canvas.draw()
            fig2.canvas.draw()

ax_button.on_clicked(on_button_click)
ax_button2.on_clicked(on_button2_click)

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()
