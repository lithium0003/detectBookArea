# distutils: define_macros=NPY_NO_DEPRECATED_API=1
# distutils: language=c++
# distutils: extra_compile_args = ["-O3","-march=native"]
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport cython
from libc.math cimport round
from libcpp.vector cimport vector

cdef cnp.uint8_t getpixel(
    vector[cnp.uint8_t]& image, cnp.float32_t x, cnp.float32_t y, int c,
    vector[cnp.uint8_t]& back,
    int im_h, int im_w, int im_c) noexcept nogil:

    cdef int x1 = <int>x
    cdef int x2 = x1 + 1
    cdef int y1 = <int>y
    cdef int y2 = y1 + 1
    cdef cnp.float32_t v1, v2, v3, v4
    cdef cnp.float32_t rx = x - x1
    cdef cnp.float32_t ry = y - y1
    if 0 <= x < im_w-1 and 0 <= y < im_h-1:
        v1 = <cnp.float32_t>image[(y1 * im_w + x1) * im_c + c]
        v2 = <cnp.float32_t>image[(y1 * im_w + x2) * im_c + c]
        v3 = <cnp.float32_t>image[(y2 * im_w + x1) * im_c + c]
        v4 = <cnp.float32_t>image[(y2 * im_w + x2) * im_c + c]
        return <cnp.uint8_t>((1 - rx)*(1 - ry)*v1 + rx*(1 - ry)*v2 + (1 - rx)*ry*v3 + rx*ry*v4)
    else:
        return back[c]

cdef cnp.uint8_t getlabel(
    vector[cnp.uint8_t]& label, cnp.float32_t x, cnp.float32_t y,
    int im_h, int im_w) noexcept nogil:

    cdef int x1 = <int>round(x)
    cdef int y1 = <int>round(y)
    if 0 <= x < im_w-1 and 0 <= y < im_h-1:
        return label[y1 * im_w + x1]
    return 0

cdef cnp.uint8_t getline(
    vector[cnp.uint8_t]& line1, cnp.float32_t x, cnp.float32_t y,
    int im_h, int im_w) noexcept nogil:

    cdef int x1 = <int>x
    cdef int x2 = x1 + 1
    cdef int y1 = <int>y
    cdef int y2 = y1 + 1
    cdef cnp.float32_t v1, v2, v3, v4
    cdef cnp.float32_t rx = x - x1
    cdef cnp.float32_t ry = y - y1
    if 0 <= x < im_w-1 and 0 <= y < im_h-1:
        v1 = <cnp.float32_t>line1[y1 * im_w + x1]
        v2 = <cnp.float32_t>line1[y1 * im_w + x2]
        v3 = <cnp.float32_t>line1[y2 * im_w + x1]
        v4 = <cnp.float32_t>line1[y2 * im_w + x2]
        return <cnp.uint8_t>((1 - rx)*(1 - ry)*v1 + rx*(1 - ry)*v2 + (1 - rx)*ry*v3 + rx*ry*v4)
    else:
        return 0

cdef void vector_dot(float& x2, float& y2, vector[cnp.float32_t]& M, float x1, float y1) noexcept nogil:
    x2 = M[0 * 3 + 0] * x1 + M[0 * 3 + 1] * y1 + M[0 * 3 + 2] * 1
    y2 = M[1 * 3 + 0] * x1 + M[1 * 3 + 1] * y1 + M[1 * 3 + 2] * 1

cdef void loop(
    vector[cnp.uint8_t]& dest1S,
    vector[cnp.uint8_t]& dest1L,
    vector[cnp.uint8_t]& dest2,
    vector[cnp.uint8_t]& dest3,
    vector[cnp.uint8_t]& dest4,
    vector[cnp.uint8_t]& dest5,
    vector[cnp.uint8_t]& image,
    vector[cnp.uint8_t]& label,
    vector[cnp.uint8_t]& line1,
    vector[cnp.uint8_t]& line2,
    vector[cnp.uint8_t]& line3,
    vector[cnp.float32_t]& matrix,
    vector[cnp.uint8_t]& back,
    int im_h, int im_w, int im_c,
    int crop_w, int crop_h) noexcept nogil:

    cdef int x1, y1
    cdef float x2, y2
    cdef float xoffset = <float>(im_w - crop_w*4) / 2
    cdef float yoffset = <float>(im_h - crop_h*4) / 2
    cdef float xoffset2 = <float>(im_w - crop_w) / 2
    cdef float yoffset2 = <float>(im_h - crop_h) / 2
    cdef float xoffsetS = <float>(crop_w*3) / 2
    cdef float yoffsetS = <float>(crop_h*3) / 2
    cdef float v1 = 0
    cdef float v2 = 0
    cdef float v3 = 0
    cdef int xt, yt
    x2 = y2 = 0

    for y1 in range(crop_h):
        for x1 in range(crop_w):
            vector_dot(x2, y2, matrix, <float>x1 + xoffset + xoffsetS, <float>y1 + yoffset + yoffsetS)
            dest1S[(y1 * crop_w + x1) * im_c + 0] = getpixel(image, x2, y2, 0, back, im_h, im_w, im_c)
            dest1S[(y1 * crop_w + x1) * im_c + 1] = getpixel(image, x2, y2, 1, back, im_h, im_w, im_c)
            dest1S[(y1 * crop_w + x1) * im_c + 2] = getpixel(image, x2, y2, 2, back, im_h, im_w, im_c)

    for y1 in range(crop_h):
        for x1 in range(crop_w):
            vector_dot(x2, y2, matrix, <float>x1 * 4 + xoffset, <float>y1 * 4 + yoffset)
            dest1L[(y1 * crop_w + x1) * im_c + 0] = getpixel(image, x2, y2, 0, back, im_h, im_w, im_c)
            dest1L[(y1 * crop_w + x1) * im_c + 1] = getpixel(image, x2, y2, 1, back, im_h, im_w, im_c)
            dest1L[(y1 * crop_w + x1) * im_c + 2] = getpixel(image, x2, y2, 2, back, im_h, im_w, im_c)

    for y1 in range(crop_h//4):
        for x1 in range(crop_w//4):
            vector_dot(x2, y2, matrix, <float>x1*4 + xoffset2, <float>y1*4 + yoffset2)
            dest2[y1 * crop_w//4 + x1] = getlabel(label, x2, y2, im_h, im_w)
            v1 = v2 = v3 = 0
            for yt in range(y1*4, y1*4+4):
                for xt in range(x1*4, x1*4+4):
                    vector_dot(x2, y2, matrix, <float>xt + xoffset + xoffsetS, <float>yt + yoffset + yoffsetS)
                    v1 += <float>getline(line1, x2, y2, im_h, im_w)
                    v2 += <float>getline(line2, x2, y2, im_h, im_w)
                    v3 += <float>getline(line3, x2, y2, im_h, im_w)
            dest3[y1 * crop_w//4 + x1] = <cnp.uint8_t>(v1 / 16)
            dest4[y1 * crop_w//4 + x1] = <cnp.uint8_t>(v2 / 16)
            dest5[y1 * crop_w//4 + x1] = <cnp.uint8_t>(v3 / 16)

cdef void transform_color(
    cnp.uint8_t &r,
    cnp.uint8_t &g,
    cnp.uint8_t &b,
    float brightness_factor,
    float contrast_factor,
    float saturation_factor,
    float hue_factor) noexcept nogil:

    cdef float r0 = <float>r / 255
    cdef float g0 = <float>g / 255
    cdef float b0 = <float>b / 255

    cdef float max_v = max(max(r0, g0), b0)
    cdef float min_v = min(min(r0, g0), b0)

    cdef float h = max_v - min_v
    cdef float s = max_v - min_v
    cdef float v = max_v

    cdef float r1 = r0
    cdef float g1 = g0
    cdef float b1 = b0
    cdef int i = 0
    cdef float f = 0

    if brightness_factor == 0 and contrast_factor == 1 and saturation_factor == 1 and hue_factor == 0:
        return

    if h > 0:
        if max_v == r0:
            h = (g0 - b0) / h
            if h < 0:
                h += 6
        elif max_v == g0:
            h = 2 + (b0 - r0) / h
        else:
            h = 4 + (r0 - g0) / h
    h /= 6
    if max_v > 0:
        s /= max_v
    
    s *= saturation_factor
    v *= contrast_factor
    h += hue_factor
    v += brightness_factor

    if h < 0:
        h += 1

    if s > 0:
        h *= 6
        i = <int>h
        f = h - <float>i
        if i == 0:
            g1 *= 1 - s * (1 - f)
            b1 *= 1 - s
        elif i == 1:
            r1 *= 1 - s * f
            b1 *= 1 - s
        elif i == 2:
            r1 *= 1 - s
            b1 *= 1 - s * (1 - f)
        elif i == 3:
            r1 *= 1 - s
            g1 *= 1 - s * f
        elif i == 4:
            r1 *= 1 - s * (1 - f)
            g1 *= 1 - s
        elif i == 5:
            g1 *= 1 - s
            b1 *= 1 - s * f
    
    if r1 < 0:
        r1 = 0
    if r1 > 1:
        r1 = 1

    if g1 < 0:
        g1 = 0
    if g1 > 1:
        g1 = 1
    
    if b1 < 0:
        b1 = 0
    if b1 > 1:
        b1 = 1

    r = <cnp.uint8_t>(r1 * 255)
    g = <cnp.uint8_t>(g1 * 255)
    b = <cnp.uint8_t>(b1 * 255)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef distortion(
    cnp.ndarray[cnp.uint8_t, ndim=3] image,
    cnp.ndarray[cnp.uint8_t, ndim=2] label,
    cnp.ndarray[cnp.uint8_t, ndim=2] line1,
    cnp.ndarray[cnp.uint8_t, ndim=2] line2,
    cnp.ndarray[cnp.uint8_t, ndim=2] line3,
    cnp.ndarray[cnp.float32_t, ndim=2] matrix,
    cnp.ndarray[cnp.uint8_t, ndim=1] back,
    int crop_w, int crop_h,
    float brightness_factor,
    float contrast_factor,
    float saturation_factor,
    float hue_factor):

    cdef int im_h = image.shape[0]
    cdef int im_w = image.shape[1]
    cdef int im_c = image.shape[2]
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] ret_small = np.empty((crop_h, crop_w, im_c), dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] ret_large = np.empty((crop_h, crop_w, im_c), dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] ret2 = np.empty((crop_h // 4, crop_w // 4), dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] ret3 = np.empty((crop_h // 4, crop_w // 4), dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] ret4 = np.empty((crop_h // 4, crop_w // 4), dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] ret5 = np.empty((crop_h // 4, crop_w // 4), dtype=np.uint8)

    cdef vector[cnp.uint8_t] im1 = vector[cnp.uint8_t](im_h * im_w * im_c)
    cdef vector[cnp.uint8_t] im2 = vector[cnp.uint8_t](im_h * im_w)
    cdef vector[cnp.uint8_t] im3 = vector[cnp.uint8_t](im_h * im_w)
    cdef vector[cnp.uint8_t] im4 = vector[cnp.uint8_t](im_h * im_w)
    cdef vector[cnp.uint8_t] im5 = vector[cnp.uint8_t](im_h * im_w)
    cdef vector[cnp.float32_t] M = vector[cnp.float32_t](3 * 3)
    cdef vector[cnp.uint8_t] backim = vector[cnp.uint8_t](im_c)
    cdef vector[cnp.uint8_t] dest1S = vector[cnp.uint8_t](crop_h * crop_w * im_c)
    cdef vector[cnp.uint8_t] dest1L = vector[cnp.uint8_t](crop_h * crop_w * im_c)
    cdef vector[cnp.uint8_t] dest2 = vector[cnp.uint8_t](crop_h//4 * crop_w//4)
    cdef vector[cnp.uint8_t] dest3 = vector[cnp.uint8_t](crop_h//4 * crop_w//4)
    cdef vector[cnp.uint8_t] dest4 = vector[cnp.uint8_t](crop_h//4 * crop_w//4)
    cdef vector[cnp.uint8_t] dest5 = vector[cnp.uint8_t](crop_h//4 * crop_w//4)

    cdef int x1, y1, c1
    cdef float im_buf
    cdef cnp.uint8_t r,g,b

    for y1 in range(im_h):
        for x1 in range(im_w):
            for c1 in range(im_c):
                im1[(y1 * im_w + x1) * im_c + c1] = image[y1,x1,c1]
            im2[y1 * im_w + x1] = label[y1,x1]
            im3[y1 * im_w + x1] = line1[y1,x1]
            im4[y1 * im_w + x1] = line2[y1,x1]
            im5[y1 * im_w + x1] = line3[y1,x1]
    for y1 in range(3):
        for x1 in range(3):
            M[y1 * 3 + x1] = matrix[y1,x1]
    for c1 in range(im_c):
        backim[c1] = back[c1]

    with nogil:
        loop(dest1S, dest1L, dest2, dest3, dest4, dest5, im1, im2, im3, im4, im5, M, backim, im_h, im_w, im_c, crop_w, crop_h)

        for y1 in range(crop_h):
            for x1 in range(crop_w):
                r = dest1S[(y1 * crop_w + x1) * im_c + 0]
                g = dest1S[(y1 * crop_w + x1) * im_c + 1]
                b = dest1S[(y1 * crop_w + x1) * im_c + 2]
                transform_color(r, g, b, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                dest1S[(y1 * crop_w + x1) * im_c + 0] = r
                dest1S[(y1 * crop_w + x1) * im_c + 0] = g
                dest1S[(y1 * crop_w + x1) * im_c + 0] = b

                r = dest1L[(y1 * crop_w + x1) * im_c + 0]
                g = dest1L[(y1 * crop_w + x1) * im_c + 1]
                b = dest1L[(y1 * crop_w + x1) * im_c + 2]
                transform_color(r, g, b, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                dest1L[(y1 * crop_w + x1) * im_c + 0] = r
                dest1L[(y1 * crop_w + x1) * im_c + 0] = g
                dest1L[(y1 * crop_w + x1) * im_c + 0] = b

    for y1 in range(crop_h):
        for x1 in range(crop_w):
            for c1 in range(im_c):
                ret_small[y1,x1,c1] = dest1S[(y1 * crop_w + x1) * im_c + c1]

    for y1 in range(crop_h):
        for x1 in range(crop_w):
            for c1 in range(im_c):
                ret_large[y1,x1,c1] = dest1L[(y1 * crop_w + x1) * im_c + c1]

    for y1 in range(crop_h//4):
        for x1 in range(crop_w//4):
            ret2[y1,x1] = dest2[y1 * crop_w//4 + x1]

    for y1 in range(crop_h//4):
        for x1 in range(crop_w//4):
            ret3[y1,x1] = dest3[y1 * crop_w//4 + x1]

    for y1 in range(crop_h//4):
        for x1 in range(crop_w//4):
            ret4[y1,x1] = dest4[y1 * crop_w//4 + x1]

    for y1 in range(crop_h//4):
        for x1 in range(crop_w//4):
            ret5[y1,x1] = dest5[y1 * crop_w//4 + x1]

    return ret_small, ret_large, ret2, ret3, ret4, ret5