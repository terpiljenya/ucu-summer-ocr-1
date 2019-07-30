# END-TO-END pipeline for text detection and recognition
originals:
 - Text detection - [EAST](https://github.com/argman/EAST)
 - Text recognition - [CRNN](https://github.com/meijieru/crnn.pytorch)




## Download

Download pretrained models:
1. frozen tensorflow EAST model [here](https://drive.google.com/file/d/1fdb91LDIRmV-269uiP9N6MVqKGqH3Gap/view?usp=sharing) [97M]
2. pretrained CRNN English model from [here](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0) [34M]

and put it to `pretrained_models` folder

Download datasets:
* [*optional*] Small train - [here](https://drive.google.com/open?id=197fBp48WU1kFXKrXr_UI_7GiCS4CuQD0) [529M]
* Validation - [here](https://drive.google.com/file/d/1rxwGwfjNhDGpvcBr-k1eumk-_ukm7V7K) [118M]

and put it to `data` folder


## Run

* test image - `run_demo_server.py` and open `http://0.0.0.0:8769/`
* validation - `validation.py`

| | Baseline (English pretrained model)  |  Benchmark (25 epochs on 80K SynthText)|
|---|---:|---:|
|**Char precision**| 0.1569 | 0.3218  |
|**Word precision**| 0.1017 | 0.1175  |


## Usefull links

#### CRNN
* paper - https://arxiv.org/abs/1507.05717

* pytorch (YouScan port) - https://github.com/YouScan/crnn.pytorch
* pytorch chinese + generation - https://github.com/Sierkinhane/crnn_chinese_characters_rec
* tensorflow - https://github.com/MaybeShewill-CV/CRNN_Tensorflow
* keras - https://github.com/Tony607/keras-image-ocr




#### Generate synthetic dataset
* SynthText (YouScan port) - https://github.com/YouScan/SynthText
* TextRecognitionDataGenerator - https://github.com/Belval/TextRecognitionDataGenerator
* pytorch CRNN chinese + generation - https://github.com/Sierkinhane/crnn_chinese_characters_rec


#### Other OCR Links
Papers/repositories/tools about text detection and recognitions:

* https://github.com/tangzhenyu/Scene-Text-Understanding
* https://github.com/jyhengcoder/myOCR
