## Docker Files for Workout Recommender

If you are running on UCSD-DSMLP, build using the Dockerfile with the header **UCSD-DSMLP Specific Container** `dsmlp.Dockerfile` (if you don't know what this is, this doesn't apply to you!)
<br />

Otherwise, use the Ubuntu **General Use Container** `ubuntu.Dockerfile` for building the files.


### Files Included in this Folder
* Dockerfiles
* docker-init.sh (Script to Run on Container Startup)
* requirements-docker.txt (requirements.txt built for Docker requirements)

### To build Dockerfile
* DSMLP: `docker build -t workout_recommender_dsmlp . -f dsmlp.Dockerfile`
* Ubuntu: `docker build -t workout_recommender . -f ubuntu.Dockerfile`

### To run Docker Image (hosted on port 5000 locally)
* DSMLP: `docker run -it -p 5000:5000 workout_recommender_dsmlp`
* Ubuntu: `docker run -it -p 5000:5000 workout_recommender`