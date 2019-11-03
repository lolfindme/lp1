def word_count(str):
	counts=dict()
	words=str.split()
	
	for word in words:
		if word in counts:
			counts[word]+=1
		else:
			counts[word]=1
	return counts
file=open('wordcount.txt','r')
alltext=file.read()
print(word_count(alltext))
