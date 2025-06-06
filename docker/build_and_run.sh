if [ -z "$2" ]; then
    echo "this requires mounting dictory with codebase and data"
    echo "run: ./run.sh codebase-dir data-dir"
    exit 1
else
    echo "mounting codebase-directory: $1 "
    echo "mounting data directory: $2"
fi

echo "Removing actloc_challenge docker image if already exists..."
docker rm -f actloc_challenge 2> /dev/null
docker rmi -f actloc_challenge_img 2> /dev/null
docker build --tag actloc_challenge_img .

# UI permisions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:docker

# create a new container
docker run -td --privileged --net=host --ipc=host \
    --name="actloc_challenge" \
    --gpus=all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e "XAUTHORITY=$XAUTH" \
    -e ROS_IP=127.0.0.1 \
    --cap-add=SYS_PTRACE \
    -v /etc/group:/etc/group:ro \
    -v "$1":/workspace \
    -v "$2":/workspace/example_data \
    actloc_challenge_img bash