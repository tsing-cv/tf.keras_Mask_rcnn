### tf.keras mask rcnn
`tsing-cv`
2019-03-02
# 1 Dependant pkgs install
```python
conda install tensorflow-gpu
conda install keras
conda install matplotlib
conda install pydot
pip install pycocotools
pip install imgaug
pip install labelme
```
# 2 convert your labelme json to coco format
## 2.1 labelme json format
```json
{
  "version": "3.6.1",
  "flags": {},
  "shapes": [
    {
      "label": "class1",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          611,
          165
        ],
        [
          697,
          153
        ],
        [
          901,
          154
        ],
        [
          1142,
          162
        ],
        [
          1068,
          1724
        ],
        [
          834,
          1700
        ],
        [
          597,
          1699
        ],
        [
          478,
          1707
        ],
        [
          549,
          934
        ]
      ],
      "shape_type": "polygon"
    },
    {
      "label": "class2",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          1866,
          89
        ],
        [
          2116,
          124
        ],
        [
          2547,
          145
        ],
        [
          2547,
          1313
        ],
        [
          2553,
          1742
        ],
        [
          1808,
          1708
        ],
        [
          1826,
          953
        ]
      ],
      "shape_type": "polygon"
    }
  ],
  "lineColor": [
    0,
    255,
    0,
    128
  ],
  "fillColor": [
    255,
    0,
    0,
    128
  ],
  "imagePath": "1.1.jpg",
  "imageData": "/9j/4AAQSkZJRgABAQAAAQABA"(an image's md5)
}
```
## 2.2 coco json format(all annotations in one big json)
```json
{
    "info": info{
                "year": int,
                "version": str,
                "description": str,
                "contributor": str,
                "url": str,
                "date_created": datetime,
            },
    "licenses": [license{
                        "id": int,
                        "name": str,
                        "url": str,
                    }],
    "images": [image{
                    "id": int,
                    "width": int,
                    "height": int,
                    "file_name": str,
                    "license": int,
                    "flickr_url": str,
                    "coco_url": str,
                    "date_captured": datetime,
                }],
    "annotations": [annotation{
                        "id": int,
                        "image_id": int,
                        "category_id": int,
                        "segmentation": RLE or [polygon],
                        "area": float,
                        "bbox": [x,y,width,height],
                        "iscrowd": 0 or 1,
                    }],
}
```
## 2.3 How to convert your json to coco
```python
cd control/
set your label jsons path, then excute 
python labelme_to_coco.py
```
- all data will be converted and saved in Coco/
# 3 train and evaluate
## 3.1 train on your dataset
1. adjust parameters of CocoConfig for your own dataset in control/coco.py; then excute 
```python
cd control/
python coco.py train
```
2. if your you want to retrain your own trained model, just excute
```python
python coco.py train --model last
```
## 3.2 evaluate on your own dataset
```python
cd control/
python coco.py evaluate --model last
```
