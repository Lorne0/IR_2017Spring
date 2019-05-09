#!/bin/bash
if [ "$#" -eq 9 ]; then
	/usr/bin/python3 hw1.py rel_on $3 $5 $7 $9 
else
	/usr/bin/python3 hw1.py rel_off $2 $4 $6 $8 
fi
