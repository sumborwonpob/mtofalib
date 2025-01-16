WS_PATH=$(realpath .)

docker run -it --rm\
	--name=roshumble-mtofalib \
	--volume ${WS_PATH}:${WS_PATH}:rw \
	--workdir ${WS_PATH} \
	--network host \
	--ipc=host \
	roshumble-mtofalib /bin/bash