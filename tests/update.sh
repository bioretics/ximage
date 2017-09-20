#!/bin/bash

for i in imgs/*_????????*.png; do
  python ../ximage.py update metadata_template.xml $i \
    ID=$(echo $i | sed -ne 's/.*_0*\([0-9]\+\).*\.png/\1/p') \
    DATE=$(date '+%Y-%m-%dT%H:%M:%S')
done
