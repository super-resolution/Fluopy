# Docker

You can use the provided Dockerfile to set up a Docker image and run the project within a container.


## Prepare for using Docker

1) Install docker (Docker Desktop)
2) Switch to Linux containers
3) Make shared drives on your host system accessible to mount local directories:

    Docker --> Settings --> Shared Drives


## Dockerfile

The provided Dockerfile creates a uv-managed Python environment on Debian Linux
and runs the test suite when the container starts.

## Build a docker image

Download the source code in a project directory.
Make sure the .dockerignore file is present.

Enter the project directory and run the following command to build the Docker image:

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
docker run -it -v <host directory>:/shared <ImageName> bash
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
