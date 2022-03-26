import os
import shutil
import pandas as pd

answer_dir = 'answers_1'
var_txt = 'vars.txt'

vars = []
answers = []

count = 0

with open(var_txt, 'r') as f:
    line = f.readline()
    while line:
        if '__' in line:
            file_name = line.split('__')[-1].split('.')[0] + '.txt'
            var = float('0.' + line.split('__')[0])
            file_path = os.path.join(answer_dir, file_name)
            with open(file_path, 'r') as f_answer:
                answer_txt = f_answer.readline()
                answer_SR = int(answer_txt.split('Answer: ')[-1])
            vars.append(var)
            answers.append(answer_SR)
            count += 1
        else:
            file_name = line[:-1].split('.')[0] + '.txt'
            file_path = os.path.join(answer_dir, file_name)
            with open(file_path, 'r') as f_answer:
                answer_txt = f_answer.readline()
                answer_GT = int(answer_txt.split('Answer: ')[-1])
            vars.append(0)
            answers.append(answer_GT)
            count += 1
        line = f.readline()

print(count)

vars = pd.Series(vars)
answers = pd.Series(answers)

pearson = vars.corr(answers, method="pearson")
print(pearson)
spearman = vars.corr(answers, method="spearman")
print(spearman)
kendall = vars.corr(answers, method="kendall")
print(kendall)