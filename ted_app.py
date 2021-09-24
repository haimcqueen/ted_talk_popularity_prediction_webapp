import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import spacy

nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

def clean_text(text):
    '''reduce text to lower-case lexicon entry'''
    lemmas = [token.lemma_ for token in nlp(text)
              if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}]
    return ' '.join(lemmas)

def clean_tags(tags):
    tags_list = list(tags)
    for i in range(len(tags_list)):
        if tags_list[i] == ' ' and tags_list[i+1] != ' ' and tags_list[i-1] != ' ':
            tags_list[i] = '_'
    tags_list = ''.join(tags_list)
    tags_list = tags_list.lower().split()
    return tags_list


def ensemble_prediction(trans_pred_proba, tags_pred_proba):
    avg_pred = (trans_pred_proba[0] + tags_pred_proba[0]) / 2
    if max(avg_pred) == avg_pred[1]:
        return (1, avg_pred)
    else:
        return 0, avg_pred

st.write("""
# TED Talk Prediction App

This app predicts if a given TED talk will be popular or not (over 1.16 Mio views - median of our training set - over its lifetime)
based on the transcript and tags of the talk.
Therefore, the model predicts future popularity based solely on the content of the talk.




**Data:** [TED Talks dataset]("https://www.kaggle.com/rounakbanik/ted-talks") by Rounak Banik

""")

example_tags = """       sleep
    medicine
    health
    medical research
    biology
    neuroscience
    brain
    human body
    science
    memory"""

example_keynote = """
Thank you very much. Well, I would like to start with testicles.

(Laughter)

Men who sleep five hours a night have significantly smaller testicles than those who sleep seven hours or more.

(Laughter)

In addition, men who routinely sleep just four to five hours a night will have a level of testosterone which is that of someone 10 years their senior. So a lack of sleep will age a man by a decade in terms of that critical aspect of wellness. And we see equivalent impairments in female reproductive health caused by a lack of sleep.

This is the best news that I have for you today.

(Laughter)

From this point, it may only get worse. Not only will I tell you about the wonderfully good things that happen when you get sleep, but the alarmingly bad things that happen when you don't get enough, both for your brain and for your body.

Let me start with the brain and the functions of learning and memory, because what we've discovered over the past 10 or so years is that you need sleep after learning to essentially hit the save button on those new memories so that you don't forget. But recently, we discovered that you also need sleep before learning to actually prepare your brain, almost like a dry sponge ready to initially soak up new information. And without sleep, the memory circuits of the brain essentially become waterlogged, as it were, and you can't absorb new memories.

So let me show you the data. Here in this study, we decided to test the hypothesis that pulling the all-nighter was a good idea. So we took a group of individuals and we assigned them to one of two experimental groups: a sleep group and a sleep deprivation group. Now the sleep group, they're going to get a full eight hours of slumber, but the deprivation group, we're going to keep them awake in the laboratory, under full supervision. There's no naps or caffeine, by the way, so it's miserable for everyone involved. And then the next day, we're going to place those participants inside an MRI scanner and we're going to have them try and learn a whole list of new facts as we're taking snapshots of brain activity. And then we're going to test them to see how effective that learning has been. And that's what you're looking at here on the vertical axis. And when you put those two groups head to head, what you find is a quite significant, 40-percent deficit in the ability of the brain to make new memories without sleep.

I think this should be concerning, considering what we know is happening to sleep in our education populations right now. In fact, to put that in context, it would be the difference in a child acing an exam versus failing it miserably -- 40 percent. And we've gone on to discover what goes wrong within your brain to produce these types of learning disabilities. And there's a structure that sits on the left and the right side of your brain, called the hippocampus. And you can think of the hippocampus almost like the informational inbox of your brain. It's very good at receiving new memory files and then holding on to them. And when you look at this structure in those people who'd had a full night of sleep, we saw lots of healthy learning-related activity. Yet in those people who were sleep-deprived, we actually couldn't find any significant signal whatsoever. So it's almost as though sleep deprivation had shut down your memory inbox, and any new incoming files -- they were just being bounced. You couldn't effectively commit new experiences to memory.

So that's the bad that can happen if I were to take sleep away from you, but let me just come back to that control group for a second. Do you remember those folks that got a full eight hours of sleep? Well, we can ask a very different question: What is it about the physiological quality of your sleep when you do get it that restores and enhances your memory and learning ability each and every day? And by placing electrodes all over the head, what we've discovered is that there are big, powerful brainwaves that happen during the very deepest stages of sleep that have riding on top of them these spectacular bursts of electrical activity that we call sleep spindles. And it's the combined quality of these deep-sleep brainwaves that acts like a file-transfer mechanism at night, shifting memories from a short-term vulnerable reservoir to a more permanent long-term storage site within the brain, and therefore protecting them, making them safe. And it is important that we understand what during sleep actually transacts these memory benefits, because there are real medical and societal implications.

And let me just tell you about one area that we've moved this work out into, clinically, which is the context of aging and dementia. Because it's of course no secret that, as we get older, our learning and memory abilities begin to fade and decline. But what we've also discovered is that a physiological signature of aging is that your sleep gets worse, especially that deep quality of sleep that I was just discussing. And only last year, we finally published evidence that these two things, they're not simply co-occurring, they are significantly interrelated. And it suggests that the disruption of deep sleep is an underappreciated factor that is contributing to cognitive decline or memory decline in aging, and most recently we've discovered, in Alzheimer's disease as well.

Now, I know this is remarkably depressing news. It's in the mail. It's coming at you. But there's a potential silver lining here. Unlike many of the other factors that we know are associated with aging, for example changes in the physical structure of the brain, that's fiendishly difficult to treat. But that sleep is a missing piece in the explanatory puzzle of aging and Alzheimer's is exciting because we may be able to do something about it.

And one way that we are approaching this at my sleep center is not by using sleeping pills, by the way. Unfortunately, they are blunt instruments that do not produce naturalistic sleep. Instead, we're actually developing a method based on this. It's called direct current brain stimulation. You insert a small amount of voltage into the brain, so small you typically don't feel it, but it has a measurable impact. Now if you apply this stimulation during sleep in young, healthy adults, as if you're sort of singing in time with those deep-sleep brainwaves, not only can you amplify the size of those deep-sleep brainwaves, but in doing so, we can almost double the amount of memory benefit that you get from sleep. The question now is whether we can translate this same affordable, potentially portable piece of technology into older adults and those with dementia. Can we restore back some healthy quality of deep sleep, and in doing so, can we salvage aspects of their learning and memory function? That is my real hope now. That's one of our moon-shot goals, as it were.

So that's an example of sleep for your brain, but sleep is just as essential for your body. We've already spoken about sleep loss and your reproductive system. Or I could tell you about sleep loss and your cardiovascular system, and that all it takes is one hour. Because there is a global experiment performed on 1.6 billion people across 70 countries twice a year, and it's called daylight saving time. Now, in the spring, when we lose one hour of sleep, we see a subsequent 24-percent increase in heart attacks that following day. In the autumn, when we gain an hour of sleep, we see a 21-percent reduction in heart attacks. Isn't that incredible? And you see exactly the same profile for car crashes, road traffic accidents, even suicide rates.

But as a deeper dive, I want to focus on this: sleep loss and your immune system. And here, I'll introduce these delightful blue elements in the image. They are called natural killer cells, and you can think of natural killer cells almost like the secret service agents of your immune system. They are very good at identifying dangerous, unwanted elements and eliminating them. In fact, what they're doing here is destroying a cancerous tumor mass. So what you wish for is a virile set of these immune assassins at all times, and tragically, that's what you don't have if you're not sleeping enough.

So here in this experiment, you're not going to have your sleep deprived for an entire night, you're simply going to have your sleep restricted to four hours for one single night, and then we're going to look to see what's the percent reduction in immune cell activity that you suffer. And it's not small -- it's not 10 percent, it's not 20 percent. There was a 70-percent drop in natural killer cell activity. That's a concerning state of immune deficiency, and you can perhaps understand why we're now finding significant links between short sleep duration and your risk for the development of numerous forms of cancer. Currently, that list includes cancer of the bowel, cancer of the prostate and cancer of the breast. In fact, the link between a lack of sleep and cancer is now so strong that the World Health Organization has classified any form of nighttime shift work as a probable carcinogen, because of a disruption of your sleep-wake rhythms.

So you may have heard of that old maxim that you can sleep when you're dead. Well, I'm being quite serious now -- it is mortally unwise advice. We know this from epidemiological studies across millions of individuals. There's a simple truth: the shorter your sleep, the shorter your life. Short sleep predicts all-cause mortality.

And if increasing your risk for the development of cancer or even Alzheimer's disease were not sufficiently disquieting, we have since discovered that a lack of sleep will even erode the very fabric of biological life itself, your DNA genetic code. So here in this study, they took a group of healthy adults and they limited them to six hours of sleep a night for one week, and then they measured the change in their gene activity profile relative to when those same individuals were getting a full eight hours of sleep a night. And there were two critical findings. First, a sizable and significant 711 genes were distorted in their activity, caused by a lack of sleep. The second result was that about half of those genes were actually increased in their activity. The other half were decreased.

Now those genes that were switched off by a lack of sleep were genes associated with your immune system, so once again, you can see that immune deficiency. In contrast, those genes that were actually upregulated or increased by way of a lack of sleep, were genes associated with the promotion of tumors, genes associated with long-term chronic inflammation within the body, and genes associated with stress, and, as a consequence, cardiovascular disease. There is simply no aspect of your wellness that can retreat at the sign of sleep deprivation and get away unscathed. It's rather like a broken water pipe in your home. Sleep loss will leak down into every nook and cranny of your physiology, even tampering with the very DNA nucleic alphabet that spells out your daily health narrative.

And at this point, you may be thinking, "Oh my goodness, how do I start to get better sleep? What are you tips for good sleep?" Well, beyond avoiding the damaging and harmful impact of alcohol and caffeine on sleep, and if you're struggling with sleep at night, avoiding naps during the day, I have two pieces of advice for you.

The first is regularity. Go to bed at the same time, wake up at the same time, no matter whether it's the weekday or the weekend. Regularity is king, and it will anchor your sleep and improve the quantity and the quality of that sleep. The second is keep it cool. Your body needs to drop its core temperature by about two to three degrees Fahrenheit to initiate sleep and then to stay asleep, and it's the reason you will always find it easier to fall asleep in a room that's too cold than too hot. So aim for a bedroom temperature of around 65 degrees, or about 18 degrees Celsius. That's going to be optimal for the sleep of most people.

And then finally, in taking a step back, then, what is the mission-critical statement here? Well, I think it may be this: sleep, unfortunately, is not an optional lifestyle luxury. Sleep is a nonnegotiable biological necessity. It is your life-support system, and it is Mother Nature's best effort yet at immortality. And the decimation of sleep throughout industrialized nations is having a catastrophic impact on our health, our wellness, even the safety and the education of our children. It's a silent sleep loss epidemic, and it's fast becoming one of the greatest public health challenges that we face in the 21st century.

I believe it is now time for us to reclaim our right to a full night of sleep, and without embarrassment or that unfortunate stigma of laziness. And in doing so, we can be reunited with the most powerful elixir of life, the Swiss Army knife of health, as it were.

And with that soapbox rant over, I will simply say, good night, good luck, and above all ... I do hope you sleep well.

Thank you very much indeed.

(Applause)

Thank you.

(Applause)

Thank you so much.

David Biello: No, no, no. Stay there for a second. Good job not running away, though. I appreciate that. So that was terrifying.

Matt Walker: You're welcome. DB: Yes, thank you, thank you. Since we can't catch up on sleep, what are we supposed to do? What do we do when we're, like, tossing and turning in bed late at night or doing shift work or whatever else?

MW: So you're right, we can't catch up on sleep. Sleep is not like the bank. You can't accumulate a debt and then hope to pay it off at a later point in time. I should also note the reason that it's so catastrophic and that our health deteriorates so quickly, first, it's because human beings are the only species that deliberately deprive themselves of sleep for no apparent reason.

DB: Because we're smart.

MW: And I make that point because it means that Mother Nature, throughout the course of evolution, has never had to face the challenge of this thing called sleep deprivation. So she's never developed a safety net, and that's why when you undersleep, things just sort of implode so quickly, both within the brain and the body. So you just have to prioritize.

DB: OK, but tossing and turning in bed, what do I do?

MW: So if you are staying in bed awake for too long, you should get out of bed and go to a different room and do something different. The reason is because your brain will very quickly associate your bedroom with the place of wakefulness, and you need to break that association. So only return to bed when you are sleepy, and that way you will relearn the association that you once had, which is your bed is the place of sleep. So the analogy would be, you'd never sit at the dinner table, waiting to get hungry, so why would you lie in bed, waiting to get sleepy?

DB: Well, thank you for that wake-up call. Great job, Matt.

MW: You're very welcome. Thank you very much.
"""

st.header("Example Prediction")
st.write("""
Here we are predicting the popularity of the following TED Talk using its content:\n
[Sleep is your Superpower](https://www.ted.com/talks/matt_walker_sleep_is_your_superpower) by Matt Walker
""")
st.video('https://www.youtube.com/watch?v=5MuIMqhT8DM&t=861s')


# Prediction using the transcript
with open ('ted_X_transcript_vectorizer.pkl', 'rb') as trans_vectorizer_file:
    transcript_vectorizer = pickle.load(trans_vectorizer_file)
with open ('ted_X_transcript_selector.pkl', 'rb') as trans_selector_file:
    transcript_selector = pickle.load(trans_selector_file)
with open ('ted_transcript.pkl', 'rb') as trans_model_file:
    transcript_model = pickle.load(trans_model_file)

example_keynote = pd.Series(clean_text(example_keynote))
X_new = transcript_vectorizer.transform(example_keynote)
X_sel = transcript_selector.transform(X_new)
trans_prediction = transcript_model.predict(X_sel)
trans_pred_proba = transcript_model.predict_proba(X_sel)

# Prediction using the transcript
with open ('ted_X_tags_vectorizer.pkl', 'rb') as tags_vectorizer_file:
    tags_vectorizer = pickle.load(tags_vectorizer_file)
with open ('ted_X_tags_selector.pkl', 'rb') as tags_selector_file:
    tags_selector = pickle.load(tags_selector_file)
with open ('ted_tags.pkl', 'rb') as tags_model_file:
    tags_model = pickle.load(tags_model_file)

example_tags_clean = clean_tags(example_tags)
example_tags = pd.Series(' '.join(example_tags_clean))
X_tags = tags_vectorizer.transform(example_tags)
X_sel_tags = tags_selector.transform(X_tags)
tags_prediction = tags_model.predict(X_sel_tags)
tags_pred_proba = tags_model.predict_proba(X_sel_tags)

prediction = ensemble_prediction(trans_pred_proba, tags_pred_proba)

#prediction
popularity = np.array(['Unpopular', 'Popular'])
st. write("Our model predicts")
trans_pred_proba, tags_pred_proba
st.subheader(popularity[prediction[0]])
st.write('\n')
st.write('\n')
st.write('\n')

st.header("Let's use a TED talk of your choice!")

with st.form(key='tags'):
    tags_input = st.text_area(label='Enter the TED Talk TAGS', height=20)
    submit_button = st.form_submit_button(label='Submit')

st.write('These are your tags:\n', clean_tags(tags_input))

with st.form(key='transcript'):
    transcript_input = st.text_area(label='Enter the TED transcript', height=100)
    submit_button = st.form_submit_button(label='Submit')

st.write('This is the transcript of your talk:\n', transcript_input[:500], '...   ')


if st.button('Predict TED Talk popularity'):
    if transcript_input is not None:
        user_keynote = pd.Series(clean_text(transcript_input))
        X_new_user = transcript_vectorizer.transform(user_keynote)
        X_sel_trans_user = transcript_selector.transform(X_new_user)
        user_trans_pred_proba = transcript_model.predict_proba(X_sel_trans_user)

    if tags_input is not None:
        user_tags = clean_tags(tags_input)
        user_tags = pd.Series(' '.join(user_tags))
        X_tags_user = tags_vectorizer.transform(user_tags)
        X_sel_tags_user = tags_selector.transform(X_tags_user)
        user_tags_pred_proba = tags_model.predict_proba(X_sel_tags_user)

    try:
        user_trans_pred_proba, user_tags_pred_proba
        user_prediction = ensemble_prediction(user_trans_pred_proba, user_tags_pred_proba)
        st. write("For your TED talk of choice our model predicts")
        st.subheader(popularity[user_prediction[0]])
    except:
        st.write('There is something wrong with your input.\nPlease try again')
