#!bin/bash


s="aaaa23"


l=$(echo $s | grep -P '\d+' -o)
echo $l
