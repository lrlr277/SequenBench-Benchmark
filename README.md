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
    - [Results](#4-Results)
        - [Main Experiment](#Main-Experiment)
        - [Circular Experiment](#Circular-Experiment)
        - [Error](#Error)
    - [Appendix](#5-Appendix)
        - [case](#case)
        - [hyperparameters](#hyperparameters)
        - [prompt](#prompt)
    - [License](#6-License)

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

<br>You can process the results of model inference through the code we provide to calculate overall accuracy,the accuracy of
each physical quantity category, overall P, R, F1 indicators,. We integrate the calculation process into the Python
files in the directory **_(Code/evaluation)_**:

```
python metrics.py
```

## 4 Results

<br>All open-source models, models, and human test results are placed in the **_(results)_**:

### Main Experiment
<br>The experimental data presented in Tables 2 and 3 were all collected from the main experiment：

 **1** The test results of the main experiment for the open-source are stored in the **_(results/base)_**

```
deepseek.jsonl
intern_14b.jsonl
intern_38b.jsonl
intern_8b.jsonl
janus.jsonl
llama.jsonl
llava.jsonl
minicpm.jsonl
mplug.jsonl
phi.jsonl
qwen_27b.jsonl
qwen_9b.jsonl
```
**2** The test results of the main experiment for the closed-source are stored in the **_(results/close)_**

```
gemini_0shot.jsonl
gemini_1shot.jsonl
gemini_2shot.jsonl
gemini_3shot.jsonl
gpt5_0shot.jsonl
gpt5_1shot.jsonl
gpt5_2shot.jsonl
gpt5_3shot.jsonl
```

### Circular Experiment
<br>To verify whether the model truly understands the sorting problem, we designed a cyclic validation experiment, and the experimental results are located in the **_(results/circular)_** folder，The experimental results are stored in the **_(results/circular/base)_** folder, with the results of open-source models placed in the /base subfolder and those of closed-source models placed in the **_(results/circular/close)_** folder

### Error
<br>We have selected two models, qwen and gemini, for error analysis, and the relevant error cases are placed in the **_(results/bad_case)_** folder

## 5 Appendix
<br>We have placed cases of different types of problems, model hyperparameters, and model prompts in a PDF file located in the **_(appenix)_** folder
### case
<br>In Section 3.2, we predefine four question types for  the convenience of annotation, with an example of each type provided in Figure 5.For details, see the file  **_(appenix)_**:
```
appendix.pdf
```
### hyperparameters
<br>The parameter settings of the open-source MLLMs used in Table 3 are provided in Table 7.For details, see the file  **_(appenix)_**:
```
appendix.pdf
```
### prompt
<br>In addition, the task prompts used for the MLLMs are listed in Table 8.For details, see the file  **_(appenix)_**:
```
appendix.pdf
```
## 6 License

<br>This project is licensed under the [Apache-2.0 License](LICENSE).
