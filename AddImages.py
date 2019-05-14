import numpy as np
import cv2

overlap_in_percent = 120

# background=retval (RGB)
# foreground=retval (RGBA, width=height)
# rectangle=(x, y, w, h)


def add_image(background, foreground, rectangle):
    x, y, w, h = rectangle

    # масштабирование
    scale = (max(w, h) * overlap_in_percent) / (foreground.shape[0] * 100)
    # print("scale {}".format(scale))
    foreground = cv2.resize(foreground, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    back_height, back_width, bach_chanel = background.shape
    # обрезаю смаил, если он не влезает на картинку
    for_width = min(foreground.shape[0], back_width - x)
    for_height = min(foreground.shape[1], back_height - y)
    foreground = foreground[:for_height, :for_width]

    fore_alpha = foreground[:, :, 3]
    fore_alpha[fore_alpha > 0] = 1
    fore_mask = np.array([[[i, i, i] for i in line] for line in fore_alpha])
    # fore_mask = np.stack((fore_alpha, fore_alpha, fore_alpha))
    fore = np.ma.array(foreground[:, :, :3], mask=np.logical_not(fore_mask), fill_value=0).filled()

    back_alpha = np.zeros((back_height, back_width))
    back_alpha[y:y + for_height, x:x + for_width] = fore_alpha

    back_mask = np.array([[[i, i, i] for i in line] for line in back_alpha])
    back = np.ma.array(background, mask=back_mask, fill_value=0).filled()
    back[y:y + for_height, x:x + for_width] += fore
    return back


if __name__ == "__main__":
    # f = np.array([
    #     [[10, 10, 10, 0], [10, 10, 10, 100]],
    #     [[10, 10, 10, 120], [10, 10, 10, 255]]
    # ])
    #
    # b = np.array([
    #     [[222, 222, 222], [222, 222, 222], [222, 222, 222], [222, 222, 222]],
    #     [[222, 222, 222], [222, 222, 222], [222, 222, 222], [222, 222, 222]],
    #     [[222, 222, 222], [222, 222, 222], [222, 222, 222], [222, 222, 222]],
    #     [[222, 222, 222], [222, 222, 222], [222, 222, 222], [222, 222, 222]]
    # ])
    #
    # # НЕНАВИЖУ, СУКА, НАСТРАИВАТЬ ЭТО ДЕГЕНАТИВНОЕ ОКРУЖЕНИЕ!
    # # У МЕНЯ НЕ РАБОТАЕТ IMSHOW() ПРОСТО ПОТОМУ ЧТО
    # f_im = cv2.imread("example.png", cv2.IMREAD_UNCHANGED)
    # print(f_im)
    # print("---------object-----------")
    # obj = add_image(b, f_im, (0, 0, 2, 2))
    # print(obj)
    # cv2.imshow("fine!", obj)
    # cv2.waitKey()

    smile = cv2.imread('smile.png')
    cv2.imshow('smile', smile)
    picture = cv2.imread('picture.jpg')
    cv2.imshow('picture', picture)

    cv2.imshow('res', add_image(picture, smile, (0, 0, 100, 80)))
