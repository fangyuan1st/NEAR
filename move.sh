#!/bin/bash
cat run.sh
echo "Enter name:"
read name
mv summary $name
mv log $name
cp run.sh $name
cp -r model $name
echo "Done"
