#!/bin/bash

# GitHub CLI api
# https://cli.github.com/manual/gh_api

longVersionName=$(git -C . describe --tags --long --dirty | tr -d '\n')
echo $longVersionName

fullVersionTag=$(echo $longVersionName | cut -d '-' -f 1)
echo $fullVersionTag
commitCount=$(echo $longVersionName | cut -d '-' -f 2)
echo $commitCount
gitSha=$(echo $longVersionName | cut -d '-' -f 3)
echo $gitSha
dirty=$(echo $longVersionName | cut -d '-' -f 4)

IFS='.' read -ra versionArray <<< "$fullVersionTag"
versionMajor=${versionArray[0]}
versionMinor=${versionArray[1]}
versionPatch=${versionArray[2]}

echo "Hello"
echo $dirty
 
#gh auth login -p ssh -h github.com
#gh release -d --generate-notes -R oist/smartphone_robot_object_detection
