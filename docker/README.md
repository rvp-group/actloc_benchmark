### Docker build & run
Build and run docker, this will mount two directories inside the containes, one with the codabase and one with data (this assumes you are inside `docker` folder):
```
chmod +x build_and_run.sh
./build_and_run.sh path-to-codebase path-to-data
```

Container should be up and running, you just log:
```
docker exec -it actloc_challenge /bin/bash
```