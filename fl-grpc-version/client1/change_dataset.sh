#!/bin/bash


s=$HOSTNAME
l=$(echo $s | grep -P '\d+' -o)
l=`expr $l + 1`
a="dataset = 'mnist'"
b="dataset = 'cifar'"
sed -i "s/$a/$b/" client.py
sed -i "s/$a/$b/" client_stream.py
