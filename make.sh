#!/bin/bash


container_name=fastApi
image_name=ai_server
docker stop $container_name
docker rm $container_name
docker rmi $image_name

#docker builder builder prune # for cache

docker build -t $image_name .
docker run -d --name $container_name -p 10100:8000 $image_name


# ## testing
# img_id=$(docker ps | grep biber | awk '{print $1}')
# docker exec -i $img_id /bin/bash -c "cd server && pytest"
