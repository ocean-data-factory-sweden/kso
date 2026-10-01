#!/bin/bash

f=playday_data.zip

if [[ -f "$f" ]]; then
    echo "Data already downloaded. Not doing anything."
    exit 0
fi

echo -n "Downloading..." && \
curl -sL "https://www.dropbox.com/scl/fo/3moo7nw5ab52qqsypmlx6/AFG5CVzmQctYGGKRfbYUSSM?rlkey=yntdqhn14fsth86b5xwu4f9bz&st=m14ospjo&dl=1" -o $f && \
echo " OK!" && \
echo -n "Extracting..." && \
unzip -q $f -x "/" && \
echo " OK!"
