if __name__ == '__main__':
    fin = open('Training_Shuffled_Dataset.txt', 'r')
    fout1 = open("from_train.txt", 'w')
    fout2 = open("to_train.txt", 'w')
    for line in fin.readlines():
        sents = line.strip().split('\t')
        fout1.write(sents[0]+'\n')
        fout2.write(sents[1]+'\n')
        fout1.write(sents[1]+'\n')
        fout2.write(sents[2]+'\n')
    fin.close()
    fout1.close()
    fout2.close()

    fin = open('Validation_Shuffled_Dataset.txt', 'r')
    fout1 = open("from_dev.txt", 'w')
    fout2 = open("to_dev.txt", 'w')
    for line in fin.readlines():
        sents = line.strip().split('\t')
        fout1.write(sents[0]+'\n')
        fout2.write(sents[1]+'\n')
        fout1.write(sents[1]+'\n')
        fout2.write(sents[2]+'\n')
    fin.close()
    fout1.close()
    fout2.close()