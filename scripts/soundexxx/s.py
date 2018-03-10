#!/bin/python
import fuzzy

wf = open("test_sents.txt", 'r')
#lf = open("../hinglishData/lang_ids.txt", 'r')
sf = open("test_soundex.txt", 'w')
wlines = wf.readlines()
#llines = lf.readlines()

soundex = fuzzy.Soundex(4)
for i in range(len(wlines)):
    #wlines[i] = "leather bag , belt aur shoe dull dikhne lage ho , to oon par kela ka chhilka ragadne se unme chamak aa jati hain"
    x = wlines[i].strip().split()
    sx = []
    for el in x:
        sxcode = "x_x"
        try:
            sxcode = soundex(el)
	    if sxcode.strip() == " " or sxcode.strip() == "":
                sxcode = "x_x"
        except:
            sxcode = "x_x"
        sx.append(sxcode)
        #print el
        #print sxcode
        #raw_input()
    sx = " ".join(sx)
    sf.write(sx+"\n")
    #print wlines[i].strip()
    #print sx
    #print len(x)
    #print len(sx.strip().split())
    #raw_input()

