#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pydub import AudioSegment


# In[3]:


for filename in os.listdir("clips"):
    name=filename.split(".")[0]
    print("clips/"+filename)

    audioseg = AudioSegment.from_mp3("clips/"+filename)
    audioseg.export("newdata/"+name+".wav", format='wav')


# In[ ]:




