# DSC180B-Project
The data we have are grabbed from the /teams directory: malware and popular-apps.

The purpose is to perform search on each software folder to find its smali files
and perform method-call analysis to build markov chain and get holistic
information of specific software and finally build a improved MAMADroid to
classify specific ware to be benign or malware.

It consists process_smali() to parse smali file and generate call-analysis.

To run it, execute python run.py <targets>.
Targets including 'feature', 'model', 'analysis', 'test'

### Responsibilities

* Jian Jiao developed code which parses content, generates features,
  builds model, perform analysis, improve model, generate results.
* Zihan Qin developed report and help partner to test code and debug.

### Project Webpage
https://kamui-jiao.github.io/DSC180B-Page/
