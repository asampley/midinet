#!/bin/bash

if [[ $# < 1 ]]; then
	DIR='songs/'
else
	DIR=$1
fi

timidity -EI6 ${DIR}/$(ls ${DIR} | sort -h | tail -n 1)
