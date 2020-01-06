from utils import *

# part 3
#datasets = ['mc', 'rg', 'wordsim']
#
#for data in datasets:
#    print('processing dataset: %s' % data)
#    with open('datasets/%s.csv' % data, newline='') as csvfile:
#        contents = list(csv.reader(csvfile, delimiter=';'))
#
#    #Change max number of results from DatamuseAPI from here
#    sim_cal = np.array(words_similarity_dataset(contents, max_num=100))
#    sim_ref = np.array(contents)[:,2].astype(float)
#    corr = pearson_correlation(sim_cal,sim_ref)
#
#    with open('results.txt', 'a') as resfile:
#        resfile.write('pearson correlation in dataset [%s] for our methods is %f\n' % (data, corr)) 

# part 5
with open('datasets/stss-131.csv', newline='') as csvfile:
    contents = list(csv.reader(csvfile, delimiter=';'))

sim_cal = sentence_similarity_dataset(contents)

with open('sentence_similarity.txt', 'a') as simfile:
    for i, pair in enumerate(contents):
        simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i] * 4))

sim_ref = np.array(contents)[:,2].astype(float) / 4.0
corr = pearson_correlation(sim_cal,sim_ref)

with open('results.txt', 'a') as resfile:
    resfile.write('pearson correlation in dataset [%s] for our methods is %f\n' % ('STS-131', corr)) 

print('done')
