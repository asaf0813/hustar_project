#!/usr/bin/env bash

uwsgi --http 127.0.0.1:5000 --wsgi-file api/app.py --callable app.py --processes 5 --threads 2
