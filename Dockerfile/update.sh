#/bin/sh
docker build . -t nvcr.io/nvidian/sae/tf2:latest -t jaycase/tf2:latest
docker push nvcr.io/nvidian/sae/tf2:latest
docker push jaycase/tf2:latest
