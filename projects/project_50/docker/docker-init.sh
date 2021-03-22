#!/usr/bin/env bash
git clone https://github.com/amandashu/Workout_Recommender.git
cd Workout_Recommender
#git checkout NK_docker
echo ""
echo "*********************************"
echo "  'python run.py all' to scrape and process data,"
echo "  'python run.py test' to test, "
echo "  'python app.py' to serve website (port 5000)"
echo "*********************************"
echo ""

echo -e '{\n  "chromedriver_path" : "/usr/local/bin/chromedriver"\n}' > config/chromedriver.json
/bin/bash