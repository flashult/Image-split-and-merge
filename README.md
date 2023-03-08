# Image split and merge

## Introduction

---

This project has two code: 

cut_image.py make image split. each sub images can have Rotate, horizontal flip, vertical flip options. split images will save in '/sub' folder

merge_image.py make original image from sub images made by cut_image.py. sub images should place in '/sub' folder

Image should be not simple image. because merge algorithm is find out most similar edge between sub images.

Split 2x2 works well but 3x3, 3x4 may have bad result.

 

### Usage

---

```python
python cut_image.py --image_file_name “ex.jpg” --column_num 2 --row_num 2
```

```python
python merge_image.py --input_file_name ./sub/ --column_num 2 --row_num 2 --output_filename merged_image
```

### Example

---

2x2

![image](https://user-images.githubusercontent.com/62612606/223814357-d74d985f-8aed-4782-a927-664f3ec23256.png)

3x3

![image](https://user-images.githubusercontent.com/62612606/223814023-2af18963-7338-40e3-ad34-19e6567456a6.png)

4x3

![image](https://user-images.githubusercontent.com/62612606/223813583-b6f6b223-0888-4725-b0e7-c21189c72f9c.png)
