#/bin/sh
docker build . -t nvcr.io/nvidian/sae/tf2:latest
docker push nvcr.io/nvidian/sae/tf2:latest
