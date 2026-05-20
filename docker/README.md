# Docker

You can use the provided Dockerfiles to set up a docker image and run the project within a container.


## Prepare for using Docker

1) Install docker (Docker Desktop)
2) Switch to Linux containers
3) Make shared drives on your host system accessible to mount local directories:

    Docker --> Settings --> Shared Drives


## Dockerfiles

We provide dockerfiles for testing and deployment.

1) Run tests in a Python 3 environment based on conda or a PyPI environment on Debian Linux.
2) Run the project in a Python 3 conda environment with Jupyter lab for interactive work (Debian Linux).


## Build a docker image

Download the source code in a project directory.
Make sure the .dockerignore file is present.

Enter the project directory and run the following command to build the docker image from one of the Dockerfiles:

```
docker build -t <ImageName> -f <Dockerfile> .
```


## Start a container from the image

### Run project tests:

Run a container to just run the project tests and close afterwards:

```
docker run --rm <ImageName>
```

### Run project in an interactive environment:
	
Open a bash shell for interactive work within a container:

```
docker run -it <ImageName> bash
```

Open the shell with a host directory mounted as volume:

```
docker run -it -v <host directory>:/home/shared <ImageName> bash
```

## Clean up

Delete a container:

```
docker rm -f <container>
```

Close all containers:

```
docker rm -f $(docker ps -q)
```

To clean up your system:

```
docker system prune
```
