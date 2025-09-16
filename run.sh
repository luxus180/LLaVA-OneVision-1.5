IMAGE=$1

docker run -it --gpus all \
    --ipc host \
    --net host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /vlm:/vlm \
    -v /mnt:/mnt \
    -v /rice_vl:/rice_vl \
    -v /train_tmp:/train_tmp \
    -v /train_tmp:/train_tmp \
    --rm \
    --name $2 \
    "$IMAGE" bash -c "service ssh restart; bash"
