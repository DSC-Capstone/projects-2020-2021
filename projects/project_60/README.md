# AutoBrick: A system for end-to-end automation of building point labels to Brick turtle files

![alt text](https://github.com/Advitya17/AutoBrickify/blob/main/autobrick_workflow.png?raw=true)

## Setup

Clone the repository and cd into the root directory.

`git clone https://github.com/Advitya17/AutoBrickify` & `cd AutoBrickify`

Then run the command below to setup the tool environment.

`python run.py env-setup` (alternatively you can build with the Dockerfile!)

This'll print a message to the console at the end to confirm setup.

#### `python run.py test` (Only for 180B Submission, can ignore otherwise)

## Instructions

### Step 1
Specify your configurations in config/data-params.json. 

Detailed instructions are available in the `config/README.md` file in this repository.


### Step 2
Run the project from the root directory.

`python run.py`

Your Turtle object file (`output.ttl`) will be generated in the root directory!
