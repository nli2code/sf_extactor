import re
import  xml.dom.minidom
from bs4 import BeautifulSoup
from nltk.tree import *
import nltk.stem as ns
from stanfordcorenlp import StanfordCoreNLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
lemmatizer = ns.WordNetLemmatizer()
body_num = 0
comment_num = 1
answer_num = 3
commentbody_num = 2
answerbody_num = 4
Questions = {}
Comments = {}
Answers = {}
def vp_list(root):
	def helper(root, res):
		if type(root)==str:
			return
		if (root.label() == "VP" or root.label() == "VB" or root.label() == "VBD" or root.label() == "VBZ" or root.label() == "VBG" or root.label() == "VBN") and root.height() <= 4:
			ret = ''
			for s in root.leaves():
				if s == "'s":
					s = "is"
				if s == "'ve":
					s = "have"
				if s == "'m":
					s = "am"
				if s == "'ll":
					s = "will"
				if s == "'re":
					s = "are"
				ret += (lemmatizer.lemmatize(s, pos='v') + "_")
			res.append(ret[:-1] + ' ')
		for i in range(len(root)):
			helper(root[i], res)
	if root is None:
		return []
	result = []
	helper(root, result)
	return result
def delete_list(root):
	def helper(root):
		for i in range(len(root)-1, -1, -1):
			if type(root[i])==str:
				return
			if root[i].label() == "NP" or root[i].label() == "NN" or root[i].label() == "RB" or root[i].label() == "RBR" or root[i].label() == "RBS" or root[i].label() == "JJ" or root[i].label() == "JJR" or root[i].label() == "JJS":
				root.remove(root[i])
			else:
				helper(root[i])
	helper(root)
	return root
def extact_question(route):
	#提取问题，并建立字典，以ID序号进行问题的存储,结果保存在Questions中
	#每个字典元素格式如下：id:[问题内容，问题评论个数,评论内容id列表，回答个数，回答内容id列表]
	dom = xml.dom.minidom.parse(route)
	root = dom.documentElement
	question_rows = root.getElementsByTagName('row')
	for row in question_rows:
		Id = row.getAttribute('Id')
		Body = row.getAttribute('Body')
		CommentCount = row.getAttribute('CommentCount')
		AnswerCount = row.getAttribute('AnswerCount')
		Questions[Id] = []
		ret = ''
		robs = BeautifulSoup(str(Body), "html.parser")
		tags = robs.find_all("p")
		for tag in tags:
			if tag.string != None:
				ret += str(tag.string)
		ret = re.sub('[^a-zA-Z.?, \']','',ret)
		if len(ret) > 0:
			treestring=nlp.parse(ret)#依存句法分析
			tree=Tree.fromstring(treestring)
			tree = delete_list(tree)
			result = vp_list(tree)
			Questions[Id].append(result)
		else:
			Questions[Id].append([])
		Questions[Id].append(CommentCount)
		Questions[Id].append([])#评论id列表
		Questions[Id].append(AnswerCount)
		Questions[Id].append([])#回答id列表
def extact_answers(route):
	#提取答案内容，保存在Answers中
	#答案格式如下id:[内容，评论个数，评论Id列表]
	dom = xml.dom.minidom.parse(route)
	root = dom.documentElement
	answer_rows = root.getElementsByTagName('row')
	for row in answer_rows:
		Id = row.getAttribute('Id')
		Body = row.getAttribute('Body')
		CommentCount = row.getAttribute('CommentCount')
		ParentId = row.getAttribute('ParentId')
		if ParentId in Questions:
			Questions[ParentId][answerbody_num].append(Id)
		Answers[Id] = []
		ret = ''
		robs = BeautifulSoup(str(Body), "html.parser")
		tags = robs.find_all("p")
		for tag in tags:
			if tag.string != None:
				ret += str(tag.string)
		ret = re.sub('[^a-zA-Z.?, \']','',ret)
		if len(ret) > 0:
			treestring=nlp.parse(ret)#依存句法分析
			tree=Tree.fromstring(treestring)
			tree = delete_list(tree)
			result = vp_list(tree)
			Answers[Id].append(result)
		else:
			Answers[Id].append([])
		Answers[Id].append(CommentCount)
		Answers[Id].append([])
def extact_comments(route):
	#提取问题的评论内容,保存在Comments中
	#每个结构是id:[内容]
	dom = xml.dom.minidom.parse(route)
	root = dom.documentElement
	comment_rows = root.getElementsByTagName('row')
	for row in comment_rows:
		Id = row.getAttribute('Id')
		Body = row.getAttribute('Text')
		PostId = row.getAttribute('PostId')
		if PostId in Questions:
			Questions[PostId][commentbody_num].append(Id)
		elif PostId in Answers:
			Answers[PostId][commentbody_num].append(Id)
		Comments[Id] = []
		ret = ''
		robs = BeautifulSoup(str(Body), "html.parser")
		tags = robs.find_all("p")
		for tag in tags:
			if tag.string != None:
				ret += str(tag.string)
		ret = re.sub('[^a-zA-Z.?, \']','',ret)
		if len(ret) > 0:
			treestring=nlp.parse(ret)#依存句法分析
			tree=Tree.fromstring(treestring)
			tree = delete_list(tree)
			result = vp_list(tree)
			Comments[Id].append(result)
		else:
			Comments[Id].append([])
def store_in_file():
	for i in Questions:
		for j in Questions[i][body_num]:
			print(j,file = out, end = '')
		for j in Questions[i][answerbody_num]:
			for k in Answers[j][body_num]:
				print(k,file = out, end = '')
		print(file=out)
def lda_train(route):
	def print_top_words(model, feature_names, n_top_words):
		#打印每个主题下权重较高的term
		for topic_idx, topic in enumerate(model.components_):
			print("Topic #%d:" % topic_idx)
			print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
		print()
		#打印主题-词语分布矩阵
		print(model.components_)
	#开始统计词频并得到词频矩阵
	tf_vectorizer = CountVectorizer(stop_words='english')
	doclist = []
	with open(route, 'r') as f:
		for line in f.readlines():
			if line != '':
				doclist.append(line.strip())
	cntTf = tf_vectorizer.fit_transform(doclist)
	# print(cntTf)
	#调参
	# parameters = {'learning_method':['online'], 
	# 			'n_components':range(20, 100, 10),
	# 			'learning_offset':[2,10,15],
	# 			'perp_tol': (0.001, 0.01, 0.1),
	# 			'max_iter':[10,100]}
	#lda = LatentDirichletAllocation()
	#gri = GridSearchCV(estimator=lda, param_grid=parameters,cv = 3,verbose = 1)
	lda = LatentDirichletAllocation(n_components=20, learning_method = 'online',max_iter = 100,verbose = 1)#n_components即为主题数
	#gri.fit(cntTf)
	#lda = gri.best_estimator_
	docres = lda.fit(cntTf)
	n_top_words=20
	tf_feature_names = tf_vectorizer.get_feature_names()
	print_top_words(lda, tf_feature_names, n_top_words)
	print("perplexity:",lda.perplexity(cntTf))
if __name__ == "__main__":
	# #以下是第一次运行时提取文本信息并分词，第二次运行时可以注释掉，需要设置好路径
	out = open("words.txt","w+",encoding = "utf-8")
	nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')
	print("开始提取Questions部分……")
	extact_question('Questions.xml')#提取问题部分
	print("开始提取Answers部分……")
	extact_answers('Answers.xml')#提取答案部分
	store_in_file()
	out.close()
	# #得到文本文件后可以进行训练
	print("开始训练模型……")
	lda_train("words.txt")