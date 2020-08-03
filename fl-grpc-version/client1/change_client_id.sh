#!/bin/bash


s=$HOSTNAME
l=$(echo $s | grep -P '\d+' -o)
l=`expr $l + 1`
a="client_id = 1"
b="client_id = $l"
sed -i "s/$a/$b/" client.py
sed -i "s/$a/$b/" client_stream.py
