# hindroid_replication
## HinDroid: An Intelligent Android Malware Detection System. Based on Structured Heterogeneous Information Network.
# Malware Detection Using API Relationships + Hindroid
# HOW TO RUN 
`python run.py test`
+ `requirements.txt`

By July 2020, Android OS is still a leading mobile operating system that holds 74.6% of market share worldwide, attracting numerous crazy cyber-criminals who are targeting at the largest crowd.¹ Also, due to its open-resource feature and flexible system updating policy, it is 50 times more likely to get infected compared to ios systems.² Thus, developing a strong malware detection system becomes the number one priority.

The current state of malware(malicious software) detection for the growing android OS application market involves looking solely at the API(Application Programming Interface) calls. API is a set of programming instructions that allow outside parties or individuals to access the platform or the application.³ A good and daily example will be the login options displaying on the app interface like “Login with Twitter”.4 Malwares can collect personal information easily from APIs, so analyzing APIs is a critical part of identifying malwares.

Hindroid formulates a meta-path based approach to highlight relationships across API calls to aggregate similarities and better detect malware.Individual APIs appearing in the ransomware could be harmless, but the combination of them could indicate “this ransomware intends to write malicious code into system kernel.”5 You wouldn’t want to see a group of “write”, “printStackTrace”, and “load” APIs appearing in your app’s smali file.5


# Data Generation Process
The data generation process and its relationship to the problem (i.e. for domain problems)
The data for identifying malware is primarily the android play store, although in order to obtain the respective APK’s for these apps the data is directly downloaded from `https://apkpure.com/`.

This data is then unpackaged using the apktools library that allows us to view the subsequent smali code and app binaries.

The smali code and app binaries contain a lot of the information derived from the Java source code that allows us to map the number of API calls and the relationships between them. 

# Observed Data 
Overall, from each android app, what’s most relevant to classifying Malware vs Benign - are the API calls, code blocks(methods) and packages these API calls occur in, classified as matrices 
A, P, B, I. This form of organizing data explains the relationship between these API calls. It provides a story, more in depth, than just the sheer number of API calls per app. 

The Hindroid model observes the same relationship of data to better classify malware or not, through the relationship of the above defined matrices. Reducing the ability of apps to just add a larger number of API calls to get classified as benign. 

# Conclusion
The process of identifying the relationship of API calls, is taking the idea of the subsequent network it creates - thus to not just look at the information queried by the call but also the way it interacts with other API’s in different levels of the codebase. The applicability of this lies beyond that of malware detection in android apps, but probably in the roots of graph theory and how relationships with API calls can be better identified and mapped out to provide more insight











## Responsibilities:
## Report:
Neel Shah: Malware Detection Using API Relationships + Hindroid, Data Generation Process, Observed Data, Conclusion
Mandy Ma: Malware Detection Using API Relationships + Hindroid, Citation
## Code:
	Neel Shah: Refining codes, create repository, transfer code format to be able to run from terminal
	Mandy Ma: Code algorithms,Debug code, Refining code


## Citation
O'Dea, Published by S., and Aug 17. “Mobile OS Market Share 2019.” Statista, 17 Aug. 2020, www.statista.com/statistics/272698/global-market-share-held-by-mobile-operating-systems-since-2009/. 
Panda Security Panda Security specializes in the development of endpoint security products and is part of the WatchGuard portfolio of IT security solutions. Initially focused on the development of antivirus software. “Android Devices 50 Times More Infected Compared to IOS - Panda Security.” Panda Security Mediacenter, 14 Jan. 2019, www.pandasecurity.com/en/mediacenter/mobile-security/android-more-infected-than-ios/
App-press.com. 2020. What Is An API And SDK? - App Press. [online] Available at: <https://www.app-press.com/blog/what-is-an-api-and-sdk#:~:text=API%20%3D%20Application%20Programming%20Interface,usually%20packaged%20in%20an%20SDK.> [Accessed 31 October 2020].
“5 Examples of APIs We Use in Our Everyday Lives: Nordic APIs |.” Nordic APIs, 10 Dec. 2019, nordicapis.com/5-examples-of-apis-we-use-in-our-everyday-lives/. 
Shifu Hou, Yanfang Ye ∗ , Yangqiu Song, and Melih Abdulhayoglu. 2017. HinDroid: An Intelligent Android Malware Detection System Based on Structured Heterogeneous Information Network. In Proceedings of KDD’17, August 13-17, 2017, Halifax, NS, Canada, , 9 pages. DOI: 10.1145/3097983.3098026
