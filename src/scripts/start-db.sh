if [ -z "$(docker images -q vanguardapps/knn-mt-pg-postgres:latest 2> /dev/null)" ]; then
    docker pull vanguardapps/knn-mt-pg-postgres:latest
fi

docker-compose up -d
