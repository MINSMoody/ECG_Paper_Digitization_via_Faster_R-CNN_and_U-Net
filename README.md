# Installation

(optional) create a conda environment  
```
conda create -n myenv python=3.10
conda activate myenv
```

install requirements of the sample code: in the toplevel folder
```
pip install -r requirements.txt
```

install requirements of our code: 
```
pip install -r TeamCode/requirements.txt
```

# Run Tests
There are some relative paths in the tests. These assume you run them from the toplevel of this repo. 

In the toplevel run
```
    python -m pytest TeamCode
```

manually try the training loop as the challenge organizers do it for submissions:
```
python train_model.py -d TeamCode/tests/resources/example_data -m mymodel
python run_model.py -d TeamCode/tests/resources/example_data -m mymodel -o myoutputs
python evaluate_model.py -d TeamCode/tests/resources/example_data -o myoutputs -s myscores.csv
```

# Developing
Only change code inside the `TeamCode` folder. 

Implement of `OurDigitizationModel.run_digitization_model` in `implementation.py`. You can run `test_our_implementation` in `tests/test_end2end_models.py` (e.g. with debugging) to check our implementation. If you set up VSCode properly you should be able to step through the test with the debugger. Clemens can help. 

# Submissions
* there must be a `model` folder containing our pretrained model which will be evaluated  


## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 100 records.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir test_submissions
        user@computer:~$ cd test_submissions
        user@computer:~/test_submissions$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2024/#data). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/test_submissions$ git clone official-phase-mins-eth

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/test_submissions$ ls
         mins-eth-submissions fake_server_setup


        user@computer:~/test_submissions$ docker build -t image official-phase-mins-eth

        # run 'image' (name from above) docker container and mount folders from the parent into the model reposetory
        # you should be in the parent folder fake_server_setup

        user@computer:~/fake_server_setup$ docker run -it -v ./model:/challenge/model -v ./test_data:/challenge/test_data -v ./test_outputs:/challenge/test_outputs -v ./training_data:/challenge/training_data image bash



        root@[...]:/challenge# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py      [...]

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d test_data -m model -o test_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d test_data -o test_outputs
        [...]

        root@[...]:/challenge# exit
        Exit

# 4b Build a different Dockerfile
```bash
 docker build -t image -f official-phase-mins-eth/Dockerfile.cpu official-phase-mins-eth
 docker save -o fake_server_setup/docker_img_cpu.tar image
```

```bash
 docker build -t gpu:v1 -f official-phase-mins-eth/Dockerfile.gpu official-phase-mins-eth
 docker save -o fake_server_setup/docker_img_gpu.tar gpu:v1
```

`docker load -i docker_img_gpu.tar`

then run command with the name instead

# try on a server with gpu 
add flag 
`--gpus all`
to docker run command. maybe some other too, check the google forum