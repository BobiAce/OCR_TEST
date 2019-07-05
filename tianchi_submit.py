import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
from network import East
from predict import predict_txt
import cfg
import time
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


result_model = 'saved_model/weights_3T736.012-0.324.h5'
# image_test_dir = '/home/jinbo/PycharmPro/DF-competition/OCR_data/mtwi_2018/icpr_mtwi_task2/image_test/'
image_test_dir = 'test_data/icpr_mtwi_task2/image_test/'
txt_test_dir = 'RESULT_TXT/0628_736/'
east = East()
east_detect = east.east_network()
east_detect.load_weights(result_model)
if __name__ == '__main__':

    test_imgname_list = os.listdir(image_test_dir)
    print('found %d test images.' % len(test_imgname_list))
    for test_img_name, k in zip(test_imgname_list,
                                range(len(test_imgname_list))):
        print('num %s image: %s' % (k, test_img_name))
        img_path = os.path.join(image_test_dir, test_img_name)
        txt_path = os.path.join(txt_test_dir, test_img_name[:-4] + '.txt')
        start = time.time()
        predict_txt(east_detect, img_path, txt_path, cfg.pixel_threshold, True)
        end = time.time()
        # print(' single image spent %s ms' % (end - start) * 1000)
