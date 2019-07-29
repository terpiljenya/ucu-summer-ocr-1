#!/usr/bin/python
# encoding: utf-8

from text_detection.detecting import TextDetector
from text_detection.utils import numpy_to_json
import os
import base64
import time
from io import BytesIO


import imagehash
import requests
import tornado.web
from tornado.httpclient import AsyncHTTPClient


import cv2
import numpy as np
import uuid
import json
import io
import math

from PIL import Image, ImageFont, ImageDraw
import requests

import tornado
import tornado.ioloop
import tornado.web
import asyncio
import os

def make_app(image_detector):
    return tornado.web.Application([
        (r"/", WebPage),
        (r"/api",  TextDetectorApi, dict(detector=detector)),
        (r"/detect",      TextDetectorUi, dict(detector=detector))
    ])


AsyncHTTPClient.configure('tornado.curl_httpclient.CurlAsyncHTTPClient', max_clients=1000)
http_client = AsyncHTTPClient()


SAVE_DIR = 'static/results'

en_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

detector = TextDetector(
    detecting_model='pretrained_models/final_detection.pb',
    recognition_model='pretrained_models/crnn.pth',
    gpu_mode=False,
    gpu_id=0,
    alphabet=en_alphabet
)


class WebPage(tornado.web.RequestHandler):
    def get(self):
        self.render('templates/index.html', has_result=False)
    def options(self):
        self.set_status(200)
    def head(self):
        self.set_status(200)


class TextDetectorApi(tornado.web.RequestHandler):
    def initialize(self, detector):
        self.detector = detector

    async def get(self):
        '''api get'''
        imageurl = self.get_query_argument('imageurl')
        try:
            response = await http_client.fetch(imageurl)
            image_buffer = response.buffer
            pil_image = Image.open(image_buffer).convert('RGB')
        except Exception as err:
            self.set_status(400)
            self.write({'status': 400, 'message': 'Cannot open image from URL - ' + imageurl})
            return

        result = self.detector.predict(pil_image)
        self.write(numpy_to_json({"rez": result['words'],
                                  'params': result['rtparams'],
                                  'timing': result['timing']}))

    async def post(self):
        '''api post'''
        try:
            rfile = self.request.files['image'][0]
            file_body = rfile['body']
            pil_image = Image.open(BytesIO(file_body))
        except Exception as err:
            self.set_status(400)
            self.write(
                {'status': 400, 'message': 'Cannot upload image', 'error': [type(err).__name__] + list(err.args)})
            return

        result = self.detector.predict(pil_image)
        self.write(numpy_to_json({"rez": result['words'],
                                  'params': result['rtparams'],
                                  'timing': result['timing']}))


class TextDetectorUi(tornado.web.RequestHandler):
    def initialize(self, detector):
        self.detector = detector

    async def get(self):
        imageurl = self.get_query_argument('imageurl')
        try:
            response = await http_client.fetch(imageurl)
            image_buffer = response.buffer
            bio = io.BytesIO()
            pil_image = Image.open(image_buffer).convert('RGB')
            pil_image.save(bio, format='jpeg')
            img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)

        except Exception as err:
            self.set_status(400)
            self.write({ 'status': 400, 'message': 'Cannot open image from URL - ' + imageurl })
            return


        result = self.detector.predict(pil_image)
        applied = draw_illu(img, result)
        img = cv2.cvtColor(applied, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img).convert('RGB')

        self.renderResult(image, result)

    def post(self):
        '''ui post'''
        try:
            rfile = self.request.files['image'][0]
            file_body = rfile['body']
            bio = io.BytesIO()
            pil_image = Image.open(BytesIO(file_body))
            pil_image.save(bio, format='jpeg')
            img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
        except Exception as err:
            self.set_status(400)
            self.write(
                {'status': 400, 'message': 'Cannot upload image', 'error': [type(err).__name__] + list(err.args)})
            return

        result = detector.predict(pil_image)
        print(result)
        applied = draw_illu(img, result)
        img = cv2.cvtColor(applied, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img).convert('RGB')
        self.renderResult(image, result)

    def renderResult(self, image, result):
        if 'errorMessage' not in result:
            has_result = True
            string_buf = BytesIO()
            image.save(string_buf, format='jpeg')
            image_src = 'data:image/png;base64,' + base64.b64encode(string_buf.getvalue()).decode().replace('\n', '')

        else:
            has_result = False
            image_src = None

        self.render('templates/index.html', has_result=has_result, result=result,
                    imagesrc=image_src)



def draw_illu(illu, rst):
    for i, t in enumerate(rst['polys']):
        bottomright = rst['words'][i]['bottomright']
        topleft = rst['words'][i]['topleft']
        #cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
        label = rst['words'][i]['label']
        cv2.rectangle(illu, tuple(int(x) for x in bottomright.values()), tuple(int(x) for x in topleft.values()), (35,94,240), 3)
        img_pil = Image.fromarray(illu)
        font_size = int(img_pil.size[0] / 256) * 3 + 10
        width = int(img_pil.size[0] / 256) + 2
        font = ImageFont.truetype("./OpenSans-Bold.ttf", font_size)
        draw = ImageDraw.Draw(img_pil)
        draw.text((int(topleft['x'] - 2), int(topleft['y'] - font_size - width)),  label, font = font, fill = (35,94,240,0))
        illu = np.array(img_pil)
        # cv2.putText(illu, label, (int(bottomright['x'] - 2), int(topleft['y'] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return illu


if __name__ == "__main__":
    app = make_app(detector)
    app.listen(8769)
    tornado.ioloop.IOLoop.current().start()
