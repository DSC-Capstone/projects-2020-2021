from limbic.emotion.models import LexiconLimbicModel
from limbic.emotion.nrc_utils import load_nrc_lexicon
from limbic.emotion.utils import load_lexicon
import numpy as np
import pandas as pd

def score_lexicon(ls,term,df):
    def find_word(sentence):
        words=[]
        for i in ls:
            if i in sentence:
                words.append(i)
        return words
    df[term] = df['clean_words'].apply(find_word)
    
def find_lexicon(df):
    s = "Unemploy, economy, rent, mortgage, evict, enough money, more money, pay the bills, owe, debt, make ends meet, afford, save enough, salary, wage, income, job, eviction"
    Economic_Stress = s.split(', ')
    s="alone, lonely, no one cares about me, no one cares, can’t see anyone, can’t see my, i miss my, i want to see my, trapped, i’m in a cage, lonely, feel ignored, ignoring me, ugly, rejected, avoid, avoiding me, am single, been single, quarantine, lockdown, isolation, self-isolation"
    Isolation = s.split(', ')
    Isolation=Isolation+['lone','reject']
    s = "smoke, cigarette, tobacco, wine, drink, beer, alcohol, drug, opioid, cocaine, snort, vodka, whiskey, whisky, tequila, meth"
    Substance_Use =s.split(', ')
    s="gun, pistol, revolver, semiautomatic, rifle, shoot, firearm, semi-automatic"
    Guns = s.split(', ')
    s ="divorce, domestic violence, abuse, yelling, fighting with me, we’re fighting, single mom, single dad, single parent, hit me, slapped me, fighting, fight"
    Domestic_Stress = s.split(', ')
    s = "commit suicide, jump off a bridge, i want to overdose, i’m a burden, i’m such a burden, i will overdose, thinking about overdose, kill myself, killing myself, hang myself, hanging myself, cut myself, cutting myself, hurt myself, hurting myself, want to die, wanna die, don’t want to wake up, don’t wake up, never want to wake up, don’t want to be alive, want to be alive, wish it would all end, done with living, want it to end, it all ends tonight, live anymore, living anymore, life anymore, be dead, take it anymore, end my life, think about death, hopeless, hurt myself, no one will miss me, don’t want to wake up, if i live or die, i hate my life, shoot me, kill me, suicide, no point"
    Suicidality = s.split(', ')
    s ="corona, virus, viral, covid, sars, influenza, pandemic, epidemic, quarantine, lockdown, distancing, national emergency, flatten, infect, ventilator, mask, symptomatic, epidemiolog, immun, incubation, transmission, vaccine"
    COVID19 = s.split(', ')
    ls = ["corona", "#corona", "coronavirus", "#coronavirus", "covid", "#covid", "covid19", "#covid19", "covid-19", "#covid-19", "sarscov2", "#sarscov2", "sars cov2", "sars cov 2", "covid_19", "#covid_19", "#ncov", "ncov", "#ncov2019", "ncov2019", "2019-ncov", "#2019-ncov", "pandemic", "#pandemic" "#2019ncov", "2019ncov", "quarantine", "#quarantine", "flatten the curve", "flattening the curve", "#flatteningthecurve", "#flattenthecurve", "hand sanitizer", "#handsanitizer", "#lockdown", "lockdown", "social distancing", "#socialdistancing", "work from home", "#workfromhome", "working from home", "#workingfromhome", "ppe", "n95", "#ppe", "#n95", "#covidiots", "covidiots", "herd immunity", "#herdimmunity", "pneumonia", "#pneumonia", "chinese virus", "#chinesevirus", "wuhan virus", "#wuhanvirus", "kung flu", "#kungflu", "wearamask", "#wearamask", "wear a mask", "vaccine", "vaccines", "#vaccine", "#vaccines", "corona vaccine", "corona vaccines", "#coronavaccine", "#coronavaccines", "face shield", "#faceshield", "face shields"]
    ls = [i.replace('#','') for i in ls]
    COVID19 = COVID19+ls
    lexicon_manual  ={'Economic_Stress':Economic_Stress,"Isolation":Isolation,"Substance_Use":Substance_Use,"Guns":Guns,"Domestic_Stress":Domestic_Stress,"Suicidality":Suicidality,"COVID19":COVID19}
    for i in lexicon_manual:
        score_lexicon(lexicon_manual[i],i,df)
        
def limbic_score(df):
    lexicon_fp = 'NRC-Emotion-Intensity-Lexicon-v1.txt'
    lexicon = load_lexicon(lexicon_fp,)

    terms_mapping = {'sentence': 'string'}
    lb = LexiconLimbicModel(lexicon, terms_mapping=terms_mapping)
    def get_emotion(s):
        return lb.get_sentence_emotions(s)
    def average_emotion(ls):    
        dic = {}
        for i in ls:
            if i.category in dic:
                dic[i.category].append(i.value)
            else:
                dic[i.category]=[i.value]
        for n in dic:
            dic[n] = np.mean(dic[n])
        return dic
    df['emotion_score'] = df['clean_text'].apply(get_emotion)
    df['average_score'] = df['emotion_score'].apply(average_emotion)
    return df
        
    