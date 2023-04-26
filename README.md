## Prompt4NR: Prompt Learning for News Recommendation
Source code for SIGIR 2023 paper: Prompt Learning for News Recommendation

### The Prompt4NR Framework

<p align='center'>
<img src="https://github.com/resistzzz/Prompt4NR/blob/main/Imgs/Prompt4NR.png" width='800'/>
</p>

### Directory Structure: 
12 directories correspond to 12 prompt templates three types (Discrete, Continuous, Hybrid) of templates from four perspectives (Relevance, Emotion, Action, Utility)
- Discrete-Relevance, Discrete-Emotion, Discrete-Action, Discrete-Utility
- Continuous-Relevance, Continuous-Emotion, Continuous-Action, Continuous-Utility
- Hybrid-Relevance, Hybrid-Emotion, Hybrid-Action, Hybrid-Utility

### Details of the 12 templates are provided as follows:

<img src="https://github.com/resistzzz/Prompt4NR/blob/main/Imgs/templates_table.png" />

### Dataset

The experiments are based on public dataset <a href="https://msnews.github.io/">MIND</a>, we use the small version MIND-Small.

For our paper, we have preprocessed the original dataset and store it as binary files via <a href="https://docs.python.org/3/library/pickle.html">"pickle"</a>. Even though I use ".txt" as the file extension, they are still binary files stored by pickle, you can use pickle package to directly load them, which include:

- train.txt: training set
- val.txt: validation set
- test.txt: testing set
- news.txt: containing information of all news

I have shared our preprocessed dataset on Google Drive as follows: 

<https://drive.google.com/drive/folders/1_3ffZvEPKD5deHbTU_mVGp6uEaLhyM7c?usp=sharing>

### How to Run These codes
In each directory, there is a script called ``run.sh`` that can run the codes for the corresponding template.
Take “Discrete-Relevance” template as an example, the ``run.sh`` file is shown as follows:
```
python main-multigpu.py --data_path ../DATA/MIND-Small --epochs 4 --batch_size 16 --test_batch_size 100 --wd 1e-3 --max_tokens 500 --log True --model_save True
python predict.py --data_path ../DATA/MIND-Small --test_batch_size 100 --max_tokens 500 --model_file ./temp/BestModel.pt --log True
```
- The first line is used to train the model on the training set and evaluate it on the validation set at each epoch. During this process, the model with the best performance on the validation set will be stored.
- The second line is used to evaluate the "best" model on the testing set to obtain the performance evaluation.

We implement the source code via the <a href="https://pytorch.org/tutorials/beginner/ddp_series_intro.html">Distributed Data Parallel (DDP)</a> technology provided by pytorch. Hence, our codes is a Multi-GPUs version. We encourage you to overwrite our code to obtain a Single-GPU version.

### Enviroments
- python==3.7
- pytorch==1.13.0
- cuda==116
- transformers==4.27.0

### Citation
If you use this codes, please cite our paper!
```
@article{zhang2023prompt,
  title={Prompt Learning for News Recommendation},
  author={Zhang, Zizhuo and Wang, Bang},
  journal={arXiv preprint arXiv:2304.05263},
  year={2023}
}
```
