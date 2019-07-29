import tensorflow as tf
import torch
from torch.autograd import Variable
import time
import collections
from PIL import Image
import numpy as np
import os


from .crnn import CRNN
from .utils import resize_image, sort_poly, detect, get_crop, resizeNormalize, strLabelConverter



class TextDetector:
    def __init__(self, detecting_model, recognition_model, gpu_mode, gpu_id, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):

        self.gpu_mode = gpu_mode
        self.gpu_id = gpu_id
        self.alphabet = alphabet

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        #load tensorflow model
        with tf.gfile.GFile(detecting_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")


        self.input_images = graph.get_tensor_by_name(
            'prefix/input_images:0')
        self.f_score = graph.get_tensor_by_name('prefix/feature_fusion/Conv_7/Sigmoid:0')
        self.f_geometry = graph.get_tensor_by_name(
            'prefix/feature_fusion/concat_3:0')

        #set up tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.15
        self.sess = tf.Session(graph=graph, config=config)

        #load pytorch model
        self.model = CRNN(32, 1, len(alphabet)+1, 256)
        if gpu_mode:
            self.model = self.model.cuda(gpu_id)

        recognition_model_state = torch.load(recognition_model, map_location='cpu')
        recognition_model_state = collections.OrderedDict({k.replace('module.', ''):v for k, v in recognition_model_state.items()})
        self.model.load_state_dict(recognition_model_state)
        self.converter = strLabelConverter(self.alphabet)
        self.transformer = resizeNormalize((100, 32))


    def predict(self, pil_image):
        start_time = time.time()
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])


        arr_image = np.array(pil_image)[:, :, ::-1]
        rtparams = collections.OrderedDict()
        rtparams['image_size'] = '{}x{}'.format(arr_image.shape[1], arr_image.shape[0])
        im_resized, (ratio_h, ratio_w) = resize_image(arr_image)
        rtparams['working_size'] = '{}x{}'.format(im_resized.shape[1], im_resized.shape[0])

        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])

        #text_detection
        start = time.time()
        score, geometry = self.sess.run(
            [self.f_score, self.f_geometry],
            feed_dict={self.input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        #nms
        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

        #change ratio
        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration

        polys = []
        if boxes is not None:
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                polys.append(tl)

        result = {
            'polys': polys,
            'timing': timer,
            'rtparams': rtparams
        }
        polys = [dict(x) for x in result['polys']]
        words = []
        for poly in polys:
            bboxes = {}
            bboxes['topleft'] = {'x': int(min(poly['x0'], poly['x3'])), 'y': int(min(poly['y0'], poly['y1']))}
            bboxes['bottomright'] = {'x': int(max(poly['x2'], poly['x1'])), 'y': int(max(poly['y2'], poly['y3']))}
            img_cropped = get_crop(pil_image, poly)

            if img_cropped is None:
                continue
            crop_pil = Image.fromarray(img_cropped)
            crop_pil = crop_pil.convert('L')
            crop_pil = self.transformer(crop_pil)
            if self.gpu_mode:
                crop_pil = crop_pil.cuda(self.gpu_id)
            crop_pil = crop_pil.view(1, *crop_pil.size())
            crop_pil = Variable(crop_pil)
            self.model.eval()
            preds = self.model(crop_pil)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
            bboxes['label'] = sim_pred
            bboxes['confidence'] = poly['score']
            words.append(bboxes)
        result['words'] = words
        return result
