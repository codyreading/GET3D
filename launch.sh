#!/bin/bash


# Determine project root dir programmatically.
SRC_PATH=$PWD
WORKING_DIR=$PWD
PIPELINE_VOLUME_P=/mnt/p
PIPELINE_VOLUME_X=/mnt/x
WORKING_VOLUME=/mnt/ml

DOCKER_IMAGE=get3d
DOCKER_IMAGE_FOUND=$(docker images | grep ${DOCKER_IMAGE} | wc -l)
if [[ ${DOCKER_IMAGE_FOUND} == "0" ]]; then
  echo "Docker Image ${DOCKER_IMAGE} not found. Exiting"
  exit 1
fi

CMD="docker run \
  --rm \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONPATH=${PYTHONPATH}:${WORKING_DIR} \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  --net=host \
  -u 0 \
  -v ${PIPELINE_VOLUME_P}:${PIPELINE_VOLUME_P} \
  -v ${PIPELINE_VOLUME_X}:${PIPELINE_VOLUME_X} \
  -v ${WORKING_VOLUME}:${WORKING_VOLUME} \
  -v ${SRC_PATH}:${WORKING_DIR} \
  -w ${WORKING_DIR} \
"
CMD+="-it --entrypoint bash ${DOCKER_IMAGE}"


eval $CMD

exit 0