import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms
import cv2
import time

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def predict(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    # east_detect.summary()
    img = image.load_img(img_path)
    # d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    d_wight, d_height = cfg.max_predict_img_size, cfg.max_predict_img_size
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height

    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    drawim = Image.open(img_path).convert('RGB')
    im_array = image.img_to_array(drawim)
    quad_draw = ImageDraw.Draw(drawim)

    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        px = int((j + 0.5) * cfg.pixel_size / scale_ratio_w)
        py = int((i + 0.5) * cfg.pixel_size / scale_ratio_h)
        line_width, line_color = 1, 'red'
        if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
            if y[i, j, 2] < cfg.trunc_threshold:
                line_width, line_color = 2, 'yellow'
            elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                line_width, line_color = 2, 'green'
        quad_draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                        (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                        (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                        (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                        (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                       width=line_width, fill=line_color)

    txt_items = []
    for score, geo, s in zip(quad_scores, quad_after_nms,
                             range(len(quad_scores))):
        if np.amin(score) > 0:
            if cfg.predict_cut_text_line:
                cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                              img_path, s)
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')

            quad_draw.line([tuple(rescaled_geo[0]),
                            tuple(rescaled_geo[1]),
                            tuple(rescaled_geo[2]),
                            tuple(rescaled_geo[3]),
                            tuple(rescaled_geo[0])], width=2, fill='blue')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    # im.save(os.path.splitext(img_path)[0] + '_act11.jpg')
    cv_im = cv2.cvtColor(np.asarray(drawim), cv2.COLOR_RGB2BGR)
    print(cv_im.shape[0:2])
    cv2.imshow("cv_im", cv_im)

    # if cfg.predict_write2txt and len(txt_items) > 0:
    #     with open(txt_path, 'w') as f_txt:
    #         f_txt.writelines(txt_items)

    # with Image.open(img_path) as im:
    #     im = im.convert('RGB')
    #     im_array = image.img_to_array(im.convert('RGB'))
    #     # d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
    #     d_wight, d_height = 736, 736
    #     scale_ratio_w = d_wight / im.width
    #     scale_ratio_h = d_height / im.height
    #     # im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    #     quad_im = im.copy()
    #     draw = ImageDraw.Draw(im)
    #     for i, j in zip(activation_pixels[0], activation_pixels[1]):
    #         px = int((j + 0.5) * cfg.pixel_size * (im.width / d_wight))
    #         py = int((i + 0.5) * cfg.pixel_size * (im.height / d_height))
    #         line_width, line_color = 1, 'red'
    #         if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
    #             if y[i, j, 2] < cfg.trunc_threshold:
    #                 line_width, line_color = 2, 'yellow'
    #             elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
    #                 line_width, line_color = 2, 'green'
    #         draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
    #                    (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
    #                    (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
    #                    (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
    #                    (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
    #                   width=line_width, fill=line_color)
    #     # im.save(os.path.splitext(img_path)[0] + '_act11.jpg')
    #     cv_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    #     print(cv_im.shape[0:2])
    #     cv2.imshow("cv_im", cv_im)
    #     quad_draw = ImageDraw.Draw(quad_im)
    #     txt_items = []
    #     for score, geo, s in zip(quad_scores, quad_after_nms,
    #                              range(len(quad_scores))):
    #         if np.amin(score) > 0:
    #             if cfg.predict_cut_text_line:
    #                 cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
    #                               img_path, s)
    #             rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
    #             rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
    #             txt_item = ','.join(map(str, rescaled_geo_list))
    #             txt_items.append(txt_item + '\n')
    #
    #             quad_draw.line([tuple(rescaled_geo[0]),
    #                             tuple(rescaled_geo[1]),
    #                             tuple(rescaled_geo[2]),
    #                             tuple(rescaled_geo[3]),
    #                             tuple(rescaled_geo[0])], width=2, fill='red')
    #         elif not quiet:
    #             print('quad invalid with vertex num less then 4.')
    #     # quad_im.save(os.path.splitext(img_path)[0] + '_predict11.jpg')
    #     cv_quid_im = cv2.cvtColor(np.asarray(quad_im), cv2.COLOR_RGB2BGR)
    #     cv2.imshow("cv_quid_im", cv_quid_im)
    #     if cfg.predict_write2txt and len(txt_items) > 0:
    #         with open(img_path[:-4] + '.txt', 'w') as f_txt:
    #             f_txt.writelines(txt_items)


def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):

    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    drawim = Image.open(img_path).convert('RGB')
    quad_draw = ImageDraw.Draw(drawim)
    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')

            quad_draw.line([tuple(rescaled_geo[0]),
                            tuple(rescaled_geo[1]),
                            tuple(rescaled_geo[2]),
                            tuple(rescaled_geo[3]),
                            tuple(rescaled_geo[0])], width=2, fill='blue')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    # cv_quid_im = cv2.cvtColor(np.asarray(drawim), cv2.COLOR_RGB2BGR)
    # cv2.imshow("cv_quid_im", cv_quid_im)

    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


if __name__ == '__main__':

    TEST_DIR = '/home/jinbo/PycharmPro/DF-competition/OCR_data/mtwi_2018/icpr_mtwi_task2/image_test/'
    imagelist = os.listdir(TEST_DIR)
    trained_model = 'saved_model/east_model_weights_3T736_h.h5'
    east = East()
    east_detect = east.east_network()
    # east_detect.load_weights(cfg.saved_model_weights_file_path)
    east_detect.load_weights(trained_model)

    # parser = argparse.ArgumentParser(description='predict')
    # parser.add_argument('--path', type=str, default='demo1/001.png', help='image path')
    # parser.add_argument('--threshold', default=cfg.pixel_threshold, help='pixel activation threshold')
    # opt = parser.parse_args()
    # img_path = opt.path
    # threshold = float(opt.threshold)
    # print(img_path, threshold)
    for i in range(len(imagelist)):
        img_path = TEST_DIR + imagelist[i]
        txt_path = img_path[:-4] + '.txt'
        predict(east_detect, img_path, txt_path, cfg.pixel_threshold)
        cv2.waitKey(0)
