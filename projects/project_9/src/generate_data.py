from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
import time
import subprocess
import json

if __name__ == "__main__":

    generate_data_params = json.load(open("../config/stdoan-generate-data-params.json"))

    network_stats = generate_data_params["network_stats_path"]
    interface = generate_data_params["interface"]
    playlist = generate_data_params["playlist"]
    outdir = generate_data_params["outdir"]
    resolutions = generate_data_params["resolutions"]

    PATH = "../web-drivers/chromedriver.exe"
    driver = webdriver.Chrome(PATH)
    driver_wait = WebDriverWait(driver, 30)

    def enable_video():
      
      player_state = driver.execute_script(
        "return document.getElementById('movie_player').getPlayerState();")
      print("current state before: " + str(player_state), flush=True)
      
      if player_state == -1:
        time.sleep(2)
        driver.execute_script("return document.getElementById('movie_player').playVideo();")
      
      if player_state == 2:
        print("video was paused; now playing", flush=True)
        driver.execute_script("return document.getElementById('movie_player').playVideo();")

    def skip_ad(wait_time):
      curr_time = 0
      while curr_time <= 0:

        try:
          video_ad = WebDriverWait(driver, wait_time).until(
              EC.element_to_be_clickable((By.CLASS_NAME, "ytp-ad-skip-button-container"))
          )
          video_ad.click()
          print("ad skipped successfully", flush=True)
          break

        except (NoSuchElementException, TimeoutException) as error:
          curr_time = driver.execute_script("return document.getElementById('movie_player').getCurrentTime()" )
          if curr_time > 0:
              print("no ad detected", flush=True)
              pass

    def set_quality(resolution):
      quality_changed = False

      while not quality_changed:

          try:
            ignore_exception = StaleElementReferenceException
            driver.find_element_by_css_selector('button.ytp-button.ytp-settings-button').click()
            driver.find_element_by_xpath("//div[contains(text(),'Quality')]").click()
            # hacky way of avoiding StaleElementException; not sure why the first run of xpath goes stale
            for i in range(2):
              quality = WebDriverWait(driver, 5, ignored_exceptions = ignore_exception).until(
                EC.visibility_of_element_located((By.XPATH, "//span[contains(text(), '{}')]".format(resolution))))
            quality.click()
            quality_changed = True
            print("Quality changed successfully", flush=True)

          except NoSuchElementException:
            print("Quality button not loaded yet", flush=True)
            time.sleep(10)
  
    ## go to playlist
    driver.get(playlist)
    videos = driver.find_elements_by_id("video-title")
    video_links = [link.get_attribute("href") for link in videos]

    for target_res in resolutions:

      # preset quality
      driver.get(video_links[0])
      enable_video()
      skip_ad(30)
      set_quality(target_res)

      for link in video_links:
        driver.get(link)

        ## collect data
        print("collecting data", flush=True)
        collect_data = subprocess.Popen(['python', network_stats, '-i' + interface, '-e' + target_res + '-' + link[-1] + '.csv'])
        driver.refresh()
        enable_video()
        skip_ad(30)

        buffer = 30
        total_dur = driver.execute_script("return document.getElementById('movie_player').getDuration();")
        overhead_diff = driver.execute_script("return document.getElementById('movie_player').getCurrentTime();")
        collect_dur = total_dur - overhead_diff
    
        print("video length: " + str(total_dur), flush=True)
        print("collection time: " + str(collect_dur - buffer), flush=True)

        skip_ad(collect_dur - buffer)

         # ensure that we are actually capturing the video data and not just ads
        while True:
          check_dur = driver.execute_script("return document.getElementById('movie_player').getCurrentTime();")
          print("current_time: " + str(check_dur), flush=True)
          
          # account for ads at beginning and for midroll ads
          remaining = collect_dur - check_dur
          print("remaining_time: " + str(remaining - buffer), flush=True)

          # check if majority of video has played
          if remaining >= buffer:
            skip_ad(remaining - buffer)
            total_collect = check_dur + remaining

          else:
            total_collect = check_dur + remaining
            break
        
        collect_data.terminate()
        print("total collected at " + target_res + " for " + str(total_collect), flush=True)
        print("=" * 120, flush=True)
    
    driver.close()






