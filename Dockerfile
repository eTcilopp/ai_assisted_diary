version: '3'

services:
  ai_assisted_diary_app:
    build: .
    volumes:
      - .:/app
    depends_on:
      db:
        condition: service_healthy  # Wait for the 'db' service to be healthy
    environment:
      DOCKER_CONTAINER: 1
      MYSQL_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ai_assisted_diary_db
      MYSQL_USER: admin

  db:
    image: mariadb
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ai_assisted_diary
      MYSQL_USER: admin
      MYSQL_PASSWORD: ${MYSQL_ROOT_PASSWORD}
    restart: always
    healthcheck:
        test: "mariadb $$MYSQL_DATABASE -uroot -p$$MYSQL_ROOT_PASSWORD -e 'SELECT 1;'"
        interval: 2s
        timeout: 2s
        retries: 10
    ports:
      - "3306:3306"
    volumes:
      - ./db_data:/var/lib/mysql
