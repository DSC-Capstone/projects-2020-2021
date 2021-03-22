#!/bin/bash
sudo apt-get update;
sudo apt install libgconf-2-4;
sudo chmod +x UnityHub.AppImage;
sudo apt-get install nodejs;
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash;
sudo apt-get install git-lfs;
git lfs install;
npm install;
