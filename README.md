## Extending Prompt4NR: Prompt Learning for News Recommendation
Code modified to extend the existing Prompt4NR framework. For the original code base please refer refer to: https://github.com/resistzzz/Prompt4NR

<p align='center'>
<img src="https://github.com/resistzzz/Prompt4NR/blob/main/Imgs/Prompt4NR.png" width='800'/>
</p>


### The Prompt4NR Framework - Extended

### Directory Structure: 
12 directories correspond to 12 prompt templates three types (Discrete, Continuous, Hybrid) of templates from four perspectives (Relevance, Emotion, Action, Utility)
- Discrete-Relevance, Discrete-Emotion, Discrete-Action, Discrete-Utility
- Continuous-Relevance, Continuous-Emotion, Continuous-Action, Continuous-Utility
- Hybrid-Relevance, Hybrid-Emotion, Hybrid-Action, Hybrid-Utility

In our extensions we only focus on the Discrete-Action template type.

The experiments are based on the <a href="https://recsys.eb.dk/dataset/">EBNeRD dataset</a> of the <a href="https://www.recsyschallenge.com/2024/">RecSys Challenge 2024</a>.

For our paper, we have preprocessed the original dataset to a large (~150k) and a large (~150k) subset and stored it as binary files via <a href="https://docs.python.org/3/library/pickle.html">"pickle"</a>. Even though I use ".txt" as the file extension, they are still binary files stored by pickle, you can use pickle package to directly load them, which include:

- train.txt: training set
- val.txt: validation set
- test.txt: testing set
- news.txt: containing information of all news

We have shared our preprocessed dataset on Google Drive as follows: 

<a href="https://drive.google.com/drive/folders/1QTA_LylrtF3RnOgO9JDUIKkLZG33FBAR?usp=sharing">Large (~150k)</a>
<a href="https://drive.google.com/drive/folders/1Gde-KkJc0szwSIXS6y3IfBxbyzY0yjnh?usp=sharing">Small (~12k)

### How to Run These codes
Since thise code utilizes multi-GPU we have wrote two scripts that make it possible to run it on a computer that supports
multi-GPU or on the Dutch National supercomputer hosted at SURF (if you have credentials) called Snellius <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html">Snellius</a>

In each directory, there is a script called ``run.sh`` that can run the codes for the corresponding template.
Take “Discrete-Relevance” template as an example, the ``run.sh`` file is shown as follows:
```
python main-multigpu.py --data_path ../DATA/MIND-Small --epochs 4 --batch_size 16 --test_batch_size 100 --wd 1e-3 --max_tokens 500 --log True --model_save True
python predict.py --data_path ../DATA/MIND-Small --test_batch_size 100 --max_tokens 500 --model_file ./temp/BestModel.pt --log True
```
- The first line is used to train the model on the training set and evaluate it on the validation set at each epoch. During this process, the model with the best performance on the validation set will be stored.
- The second line is used to evaluate the "best" model on the testing set to obtain the performance evaluation.


### Enviroments
To easily create an environment that supports running this code we have created a .yml file in ```recsys_gpu.yml```.

* General machine: install using e.g. using conda ```conda env create -f recsys_gpu.yml```
* Snellus: job file that creates your environment for you is called ```batch_jobs/setup-env.job```. Push it to the batch node. 

### Citation
If you use this codes, please cite the original paper!
```
@inproceedings{zhang2023prompt,
    author = {Zhang, Zizhuo and Wang, Bang},
    title = {Prompt Learning for News Recommendation},
    year = {2023},
    booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {227–237},
    numpages = {11},
    location = {Taipei, Taiwan},
    series = {SIGIR '23}
}
```
