#!/bin/bash

# Pull the latest changes from Git
git pull

# Start your application
exec "$@"