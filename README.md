# SequenBench-Benchmark

<br />
<p align="center">
  <h1 align="center"> 📐Small Sequences, Great Challenges: The Limits of MLLMs in Multimodal Sequence Reasoning </h1>
  <h3 align="center">SequenBench: A new benchmark dataset for ordering of images.</h3>

  <p align="center">  
<!--     <a href="">arxiv</a> -->
    ·
    <a href="https://github.com/lrlr277/SequenBench-Benchmark/blob/main/Dataset/data.jsonl">github</a>
    ·
    <a href="https://github.com/lrlr277/SequenBench-Benchmark/blob/main/LICENSE">license</a>
<!--     <a href="">benchmark</a> -->

</p>

## Contents

- [SequenBench](#Contents)
    - [Overview](#1-Overview)
    - [Access SequenBench](#2-Access-SequenBench)
        - [Data Split](#Data-Split)
        - [Data Format](#Data-Format)
    - [Experiment & Evaluation](#3-Experiment-and-Evaluation)
        - [Experiment](#Experiment)
        - [Evaluation](#Evaluation)
    - [License](#4-License)

## 1 Overview
**SequenBench**  is a benchmark for testing **the visual ranking ability** of multimodal large language models, consisting of 6761 images and
7261 multiple-choice questions

## 2 Access SequenBench
<br>All the  questions,options and answers are in the directory **_(Dataset)_**.
<br>All the  images are in the directory **_(Images/)_**.

### Data Split
As reported in the folloeing table, SequenBench contains 7261 samples, divided into training,dev sets, test sets
according to a 7:1:2 ratio.
<br>All the splited data sets are in the directory **_(Dataset)_**.

### Data Format

Each `jsonl` file is of the following format:

```
json
{
  "image": "3-5-0001.jpg",
  "question": "abcd represent four processes of heating water in a pan, with their temperatures labeled as a, b, c, and d respectively. If the water temperatures shown in the four figures are to be arranged in order from lowest to highest, which of the following is the correct sequence?",
  "options": [
    "A.dabc",
    "B.abdc",
    "C.adbc",
    "D.bcda"
  ],
  "answer": "C"
}
{
  "image": "5-7-0002.jpg",
  "question": "Assuming a, b, c, and d represent the ear lengths of four dogs from left to right, which of the following options correctly sorts them from shortest to longest?",
  "options": [
    "A.dabc",
    "B.abcd",
    "C.abdc",
    "D.adbc"
  ],
  "answer": "D"
}
{
  "image": "1-5-0001.jpg",
  "question": "There are four books abcd stacked together in the picture.Please sort them by thickness from smallest to largest. Which of the following options is correct?",
  "options": [
    "A.abcd",
    "B.bacd",
    "C.badc",
    "D.abdc"
  ],
  "answer": "B"
}
{
  "..."
}
```

Each line is an individual data point.
`image` denotes number of the image . `question` is the description related to image sorting, `options` is an image arrangement order or description related to arrangement
options.
<br>

## 3 Experiment and Evaluation

### Experiment

We have disclosed the inference code for the model in the directory **_(Code)_**, as well as the fine-tuning
code in the directory **_(Code/open_models/finetune)_**.
<br>

- For all 9 open-sourse MLLMs, you can execute Python files in the directory **_(Code/open_models/inference)_** 

```
nohup python deepseek.py
nohup python intern.py
nohup python Janus.py
nohup python llama.py
nohup python llava.py
nohup python minicpm.py
nohup python mplug.py
nohup python Phi.py
nohup python qwen.py
```



- For gemini3.5-flash and gpt-5.5, you can directly execute our Python file in the directory **_(Code/close_models)_** to
  perform inferencing of the zero-shot, few-shot, provided that you prepare a key:

```
python gemini.py
python gpt5.py
```

gemini3.5-flash needs to apply on the [official website](https://aistudio.google.com/app/apikey), and GPT-5.5 needs to be
purchased on the [official website](https://openai.com/).

### Evaluation

You can process the results of model inference through the code we provide to calculate overall accuracy,the accuracy of
each physical quantity category, overall P, R, F1 indicators,. We integrate the calculation process into the Python
files in the directory **_(Code/evaluation)_**:

```
python metrics.py
```

## 4 License

This project is licensed under the [Apache-2.0 License](LICENSE).
