# DSC180b-Capstone

Project repository for Recommender Systems group 3

This project is focused on creating music recommendations for users and their parents.

# HOW TO RUN

Our project's current targets are: load-data, task0, task1, task2, all, test

Our project's current config files are: test.json and run.json

### Targets

load-data: Pulls our training data from a S3 bucket where we store it and creates a new 'data'
repository to store it.

task0: Generates a list of sample parent recommendations and saves them to data/recommendations as a csv file.

task1: Generates a list of parent-user recommendations and saves them to data/recommendations as a csv file.

task2: Generates a list of user-parent recommendations base on a user's listening history. Saves
these recommendations to data/recommendations as a csv file.

all: runs load-data, task0, task1, and task2 in succession

test: Runs through the same load-data, task0, task1, task2 pipeline but uses a pre-stored user access code.
This allows us to 'test' our recommendation models without having to authenticate ourselves every
single time. The test data in this case is a user account that we have permission to read from.

### Configs

Our configuration files are relatively simple given our project's reliance on listening histories and
otherwise limited user information. The values in each file are the same, but we have created two files
so that logic can be quickly tested without having to constantly change the configuration parameters during
development.

username: The username of the Spotify account that we are creating recommendations for
parent_age: The age of the user's parent
genre: The parent's preferred genre of music
artist: The parent's preferred artist

This information would normally be provided by users through a form on our website, but for this situation
we just reference configuration files.


### IN THE FUTURE

We plan on adding a clean-data target that isolates some of the small preprocessing that our code does: dropping na values
small transformations, etc.

Instead of caching an auth_token for a specific user, we want to just directly load listening history. This would be a more
straightforward procedure, the problem is that we inevitably have to authenticate regardless of if we have our user data predownloaded or not. Spotify has another authentication flow that would better suit itself to this situation, and we will be looking into that in the future.

