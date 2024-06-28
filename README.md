## *Extending* Prompt4NR: Prompt Learning for News Recommendation
Code modified to *extend* the existing Prompt4NR framework. For the original code base please refer to: 
<a href="https://github.com/resistzzz/Prompt4NR" target="_blank" rel="noopener noreferrer">https://github.com/resistzzz/Prompt4NR</a>

<p align='center'>
<img src="https://github.com/resistzzz/Prompt4NR/blob/main/Imgs/Prompt4NR.png" width='800'/>
</p>


### The Prompt4NR Framework - *Extended*

### Directory Structure: 
12 directories correspond to 12 prompt templates three types (Discrete, Continuous, Hybrid) of templates from four perspectives (Relevance, Emotion, Action, Utility)
- Discrete-Relevance, Discrete-Emotion, Discrete-Action, Discrete-Utility
- Continuous-Relevance, Continuous-Emotion, Continuous-Action, Continuous-Utility
- Hybrid-Relevance, Hybrid-Emotion, Hybrid-Action, Hybrid-Utility

In our extensions we only focus on the Discrete-Action prompt template type. Folder containing this functionality is therefor ```Discrete-Action/```.

## Datasets - general
The experiments are based on the <a href="https://recsys.eb.dk/dataset/" target="_blank" rel="noopener noreferrer">EBNeRD dataset</a> of the <a href="https://www.recsyschallenge.com/2024/" target="_blank" rel="noopener noreferrer">RecSys Challenge 2024</a>.

For our paper, we have preprocessed the original dataset to a large (~150k) and a small (~12k) subset and stored it as binary files via <a href="https://docs.python.org/3/library/pickle.html" target="_blank" rel="noopener noreferrer">"pickle"</a>. Even though we use ".txt" as the file extension, they are still binary files stored by pickle, you can use pickle package to directly load them, which include:

- train.txt: training set
- val.txt: validation set
- test.txt: testing set
- news.txt: containing information of all news

The only file containing natural language sentences is news.txt, to run the experiments for English data therefor replace this file with the English variant while the rest remains the same. 

By default we placed these datasets at the location ```DATA/English_small```, but this folder name can be anything as long as it matches the directory path variable set in the .job or .sh files (explanation will become apparent in the **How to Run These codes** section below). 

We have shared our preprocessed dataset on Google Drive as follows: 

* <a href="https://drive.google.com/drive/folders/1QTA_LylrtF3RnOgO9JDUIKkLZG33FBAR?usp=sharing" target="_blank" rel="noopener noreferrer">Large (~150k)</a>
* <a href="https://drive.google.com/drive/folders/1Gde-KkJc0szwSIXS6y3IfBxbyzY0yjnh?usp=sharing" target="_blank" rel="noopener noreferrer">Small (~12k)</a>

## Datasets - clustering
To perform the data clustering a dataset has been constructed with the users their history the per article features in the correct datastructure (for specifications see the notebook ```history_selection/clustering.ipynb```). This dataset can be found here: <a href="https://drive.google.com/file/d/1iiO71WqTiiaIyA6UE6q0351fM_TYb9Bs/view?usp=sharing">final_cluster_data.txt</a> and should be placed in the framework as follows ```DATA/user_history_selection/final_cluster_data.txt```. With this dataset it is possible to perform the clustering yourself with the notebook called ```history_selection/clustering.ipynb```. However, to save you computation time the resulting clustered users' history are also provided such that you don't have to perform the clustering yourself and can just run the Prompt4NR framework. The clustered users' history can be found at <a href="https://drive.google.com/file/d/1FfyuF5qfj85PUleSNYM_SHKLSgl_iZyy/view?usp=sharing">```user_clustered_articles_history.pickle```</a>. And should be placed as follows for Prompt4NR to be able to run:  ```DATA/user_history_selection/user_clustered_articles_history.pickle```.

### How to Run These codes
> [!IMPORTANT]
> Guidelines based on Linux OS.

Since this code utilizes multi-GPU we have written two scripts that make it possible to run it on a computer that supports
multi-GPU or on the Dutch National supercomputer hosted at SURF (if you have credentials) called <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html" target="_blank" rel="noopener noreferrer">Snellius</a>

* **General machine:** the bash script that runs the code to train and predict is located at ```Discrete-Action/discrete-action_train_predict.sh```. Go this its directory and execute this script ```./discrete-action_train_predict.sh```.

* **Snellius:** the job file that runs the code to train and predict is located at ```batch_jobs/Discrete-Action/discrete-action_train_predict.job```. Go to its directory and push the job file to the batch node ```sbatch discrete-action_train_predict.job```.

In all files the lines to run the scripts contain flags to for the experiments settings (such as hyperparameters) but also which prompt template to experiment with. Arguments that use variables recognizable by the ```$``` sign should remain untouched as these intend to automatically generate correct path names. They do however have to be correctly initialized with the desired values (specifications in the .sh and .job files themselves). Other flags' arguments can be set as desired.

An example in the ```Discrete-Action/discrete-action_train_predict.sh``` is as follows:
```
python3 -u main-multigpu.py --cluster_data_avail True --prompt_type original --data_path $TMPDIR$DATA_SET --model_name $MODEL_NAME --epochs 3 --batch_size 16 --test_batch_size 100 --wd 1e-3 --max_tokens 500 --log True --world_size 4 --model_save True
python3 -u predict.py --cluster_data_avail True --data_path $TMPDIR$DATA_SET --model_name $MODEL_NAME --test_batch_size 100 --max_tokens 500 --model_file ./temp/$MODEL_NAME$DATA_SET/$date/BestModel.pt --log True --world_size 4
```
- The first line is used to train the model on the training set and evaluate it on the validation set at each epoch. During this process, the model with the best performance on the validation set will be stored.
- The second line is used to evaluate the "best" model on the testing set to obtain the performance evaluation.


### Enviroments
To easily create an environment that supports running this code we have created a .yml file in ```recsys_gpu.yml```.

* **General machine:** install using e.g. using conda ```conda env create -f recsys_gpu.yml```
* **Snellus: job file** that creates your environment for you is called ```batch_jobs/setup-env.job```. Go to its directory and push it to the batch node ```sbatch setup-env.job```. 

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
