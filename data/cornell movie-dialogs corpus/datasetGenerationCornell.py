if __name__ == '__main__':
	lines = {}
	delim = ' +++$+++ '

	with open('movie_lines.txt', 'rb') as f:
		for line in f.readlines():
			try:
				line = line.decode('utf-8')
				split_line = line.strip().split(delim)
				if len(split_line) == 5: # some line has length four
					lines[split_line[0]] = split_line[4] # lines: dict that maps line number to line
			except UnicodeError:
				print("String is not UTF-8")

	conversation = []
	with open('movie_conversations.txt', 'rb') as f:
		for line in f.readlines():
			try:
				line = line.decode('utf-8')
				line_list = line.strip().split(delim)[-1]
				line_list = line_list[1:-1].replace("'", "").split(', ')
				conversation.append(line_list)
			except UnicodeError:
				print("String is not UTF-8")
	
	# Generate from and to dataset
	fout1 = open("from_train.txt", 'w')
	fout2 = open("to_train.txt", 'w')
	for line_list in conversation:
		if len(line_list) >= 2:
			for i in range(len(line_list)-1):
				if line_list[i] in lines and line_list[i+1] in lines:
					fout1.write(lines[line_list[i]]+'\n')
					fout2.write(lines[line_list[i+1]]+'\n')
	fout1.close()
	fout2.close()
	