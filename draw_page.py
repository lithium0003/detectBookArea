from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import interpolate
import sys, os

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

if len(sys.argv) != 2:
    print("Usage: python draw_page.py <target_file>")
    sys.exit(1)

target_file = sys.argv[1]

im0 = Image.open(target_file).convert('RGB')
im = np.array(im0)

if os.path.exists(target_file + '.labeled.png'):
    labelim0 = Image.open(target_file + '.labeled.png').convert('L')
    labelim = np.array(labelim0)
else:
    labelim = np.zeros(im.shape[:2], dtype=np.uint8)
line_points = []
move_i = None
closed = False

cmap = plt.get_cmap('Set1')
colors = [cmap(i) for i in range(5)]
colors = [(0.0, 0.0, 0.0, 0.0)] + colors
cmap = ListedColormap(colors, name="custom")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.suptitle("0: background, 1: Main, 2: title, 3: outer, 4: figure, 5: CM\nSpace: save, c: clear, Enter: close line")
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, sharex=ax, sharey=ax)
ax.imshow(im0)
ax.imshow(labelim, alpha=0.25, cmap=cmap, vmin=0, vmax=5)
ax2.imshow(labelim, cmap=cmap, vmin=0, vmax=5)

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
        elif len(points) > 2:
            tck, u = interpolate.splprep([[p[1] for p in points], [p[0] for p in points]], k=len(points)-1, s=0)
            u = np.linspace(0,1,num=1000,endpoint=True)
            x, y = interpolate.splev(u, tck)
            xy += list(zip(x, y))

    draw.polygon(xy, outline=255, fill=255)
    return np.array(im1)

def on_press(event):
    global labelim, line_points, move_i, closed
    sys.stdout.flush()
    if event.key == ' ':
        print("Saved.")
        Image.fromarray(labelim).save(target_file + '.labeled.png')
        return

    if event.key == 'c':
        print("Deleted.")
        line_points = []
        move_i = None
        closed = False
        for c in list(ax.collections):
            c.remove()
        for c in list(ax.lines):
            c.remove()
        return

    if not closed and len(line_points) > 2 and event.key == 'enter':
        closed = True
        line_points.append([line_points[-1][-1], line_points[0][0]])

        for c in list(ax.collections):
            c.remove()
        for c in list(ax.lines):
            c.remove()

        for points in line_points:
            if closed:
                ax.scatter([p[1] for p in points], [p[0] for p in points], c='red')
            else:
                ax.scatter([p[1] for p in points], [p[0] for p in points], c='green')
        for points in line_points:
            if len(points) == 2:
                ax.plot([points[0][1], points[1][1]], [points[0][0], points[1][0]], c='blue')
            elif len(points) > 2:
                tck, u = interpolate.splprep([[p[1] for p in points], [p[0] for p in points]], k=len(points)-1, s=0)
                u = np.linspace(0,1,num=1000,endpoint=True)
                x, y = interpolate.splev(u, tck)
                ax.plot(x, y, c='blue')
        fig.canvas.draw()

        return

    if not closed:
        print("Please close the line first.")
        return

    if event.key == '0':
        labelim[drawimage() > 0] = 0
    elif event.key == '1':
        labelim[drawimage() > 0] = 1
    elif event.key == '2':
        labelim[drawimage() > 0] = 2
    elif event.key == '3':
        labelim[drawimage() > 0] = 3
    elif event.key == '4':
        labelim[drawimage() > 0] = 4
    elif event.key == '5':
        labelim[drawimage() > 0] = 5
    else:
        return

    for c in list(ax.collections):
        c.remove()
    for c in list(ax.lines):
        c.remove()
    ax.images[-1].remove()
    ax.imshow(labelim, alpha=0.25, cmap=cmap, vmin=0, vmax=5)
    ax2.clear()
    ax2.imshow(labelim, cmap=cmap, vmin=0, vmax=5)
    fig.canvas.draw()
    fig2.canvas.draw()

    line_points = []
    move_i = None
    closed = False

def onclick(event):
    global line_points, move_i, closed
    x = event.xdata
    y = event.ydata
    if x is not None and y is not None:
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
            move_i = None
            if len(line_points) < 3:
                closed = False
        elif event.dblclick:
            xi = int(x)
            yi = int(y)
            if 0 <= xi < im.shape[1] and 0 <= yi < im.shape[0]:
                if len(line_points) == 0:
                    line_points.append([(y, x)])
                elif len(line_points[-1]) == 1:
                    line_points[-1].append((y, x))
                else:
                    line_points.append([line_points[-1][-1], (y, x)])
                if len(line_points) > 2 and abs(line_points[0][0][0] - line_points[-1][-1][0]) < 5 and abs(line_points[0][0][1] - line_points[-1][-1][1]) < 5:
                    closed = True
                    line_points[-1][-1] = line_points[0][0]
                move_i = None
        else:
            for li, points in enumerate(line_points):
                for i, (px, py) in enumerate(points):
                    if abs(px - y) < 5 and abs(py - x) < 5:
                        move_i = (li, i)
                        break
                else:
                    continue
                break
            else:
                for li, points in enumerate(line_points):
                    if len(points) == 2:
                        dx = points[1][1] - points[0][1]
                        dy = points[1][0] - points[0][0]
                        if abs(dx) > abs(dy):
                            a = dy / dx
                            c = points[0][0] - a * points[0][1]
                            d = abs(a * x + c - y) / np.sqrt(a**2 + 1)
                        else:
                            a = dx / dy
                            c = points[0][1] - a * points[0][0]
                            d = abs(a * y + c - x) / np.sqrt(a**2 + 1)
                        if d < 5:
                            line_points[li].insert(1, (y, x))
                            move_i = (li, 1)
                        else:
                            continue
                        break
                    elif 2 < len(points) < 5:
                        tck, u = interpolate.splprep([[p[1] for p in points], [p[0] for p in points]], k=len(points)-1, s=0)
                        u2 = np.linspace(0,1,num=1000,endpoint=True)
                        for i, (px, py) in enumerate(zip(*interpolate.splev(u2, tck))):
                            if np.sqrt((px - x)**2 + (py - y)**2) < 5:
                                for j, u1 in enumerate(u):
                                    if i/1000.0 > u1:
                                        continue
                                    line_points[li].insert(j, (y, x))
                                    move_i = (li, j)
                                    break
                                else:
                                    line_points[li].insert(len(line_points[li]), (y, x))
                                    move_i = (li, len(line_points[li]) - 1)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                else:
                    return

        for c in list(ax.collections):
            c.remove()
        for c in list(ax.lines):
            c.remove()

        for points in line_points:
            if closed:
                ax.scatter([p[1] for p in points], [p[0] for p in points], c='red')
            else:
                ax.scatter([p[1] for p in points], [p[0] for p in points], c='green')
        for points in line_points:
            if len(points) == 2:
                ax.plot([points[0][1], points[1][1]], [points[0][0], points[1][0]], c='blue')
            elif len(points) > 2:
                tck, u = interpolate.splprep([[p[1] for p in points], [p[0] for p in points]], k=len(points)-1, s=0)
                u = np.linspace(0,1,num=1000,endpoint=True)
                x, y = interpolate.splev(u, tck)
                ax.plot(x, y, c='blue')
        fig.canvas.draw()

def onmove(event):
    global line_points, move_i, closed
    if move_i is None:
        return
    x = event.xdata
    y = event.ydata
    if x is not None and y is not None:
        li, i = move_i
        if i == 0:
            line_points[li][i] = (y, x)
            if li > 0:
                line_points[li-1][-1] = (y, x)
            elif closed:
                line_points[-1][-1] = (y, x)
        elif i == len(line_points[li]) - 1:
            line_points[li][i] = (y, x)
            if li < len(line_points) - 1:
                line_points[li+1][0] = (y, x)
            elif closed:
                line_points[0][0] = (y, x)
        else:
            line_points[li][i] = (y, x)

        for c in list(ax.collections):
            c.remove()
        for c in list(ax.lines):
            c.remove()

        for points in line_points:
            if closed:
                ax.scatter([p[1] for p in points], [p[0] for p in points], c='red')
            else:
                ax.scatter([p[1] for p in points], [p[0] for p in points], c='green')
        for points in line_points:
            if len(points) == 2:
                ax.plot([points[0][1], points[1][1]], [points[0][0], points[1][0]], c='blue')
            elif len(points) > 2:
                tck, u = interpolate.splprep([[p[1] for p in points], [p[0] for p in points]], k=len(points)-1, s=0)
                u = np.linspace(0,1,num=1000,endpoint=True)
                x, y = interpolate.splev(u, tck)
                ax.plot(x, y, c='blue')
        fig.canvas.draw()

def onrelease(event):
    global move_i
    move_i = None

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('button_release_event', onrelease)
fig.canvas.mpl_connect('motion_notify_event', onmove)
fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()
