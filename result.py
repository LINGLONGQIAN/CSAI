import pickle
import os
import pandas as pd
import re
import shutil

def find_and_process_directories(root_path, date_format):
    # Convert the provided date format to a regular expression pattern
    date_pattern = date_format.replace('%Y', r'\d{4}').replace('%m', r'\d{2}').replace('%d', r'\d{2}')\
                              .replace('%H', r'\d{2}').replace('%M', r'\d{2}').replace('%S', r'\d{2}')
    date_pattern = re.compile(date_pattern)
    
    # Walk through the directory
    j = 0
    jj = 0
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            # Check if the directory name matches the date format
            if date_pattern.match(dir):
                full_dir_path = os.path.join(root, dir)
                jj += 1
                # Check for any .pkl files in the directory
                if any(file.endswith('.pkl') for file in os.listdir(full_dir_path)):
                    j += 1
                    print(full_dir_path)
                    parent_dir = os.path.dirname(full_dir_path)
                    # Record the directory name in exp_datetime.txt in the parent directory
                    with open(os.path.join(parent_dir, "exp_datetime.txt"), "a") as log_file:
                        log_file.write(dir + "\n")
                    # Move contents of the directory to its parent directory
                    for item in os.listdir(full_dir_path):
                        shutil.move(os.path.join(full_dir_path, item), parent_dir)
                    # Remove the now-empty directory
                    os.rmdir(full_dir_path)
    print(jj)
    print(j)

# Execute the function with the given directory path and date format
find_and_process_directories(directory_path, r'%Y%m%d\.%H\.%M\.%S')

dataset = 'physionet'
log = 'log'

# Path to the root directory where the search should start
directory_path = './{}/{}/'.format(log, dataset)
# Execute the function with the given directory path and date format
find_and_process_directories(directory_path, r'%Y%m%d\.%H\.%M\.%S')

file_dir = os.path.join('results')
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
    
task_result = os.path.join(file_dir, dataset)
if not os.path.exists(task_result):
    os.makedirs(task_result)
    os.makedirs(task_result + '/valid/')

exps = os.listdir(directory_path)

for model_name in exps:
    try:
        path = 'directory_path/{}/'.format(model_name)
        results = pd.DataFrame()
        j = 0
        for i in os.listdir(path):
            try:
                result = pickle.load(open(path + i + '/kfold_best.pkl', 'rb'))
                subresults = pd.DataFrame()
                for key, value in result.items():
                    if 'bets_valid' in key:
                        value['model']=i
                        value['fold'] = key
                        subresults = pd.concat([subresults, pd.DataFrame([value])])
                overall = subresults[subresults.columns.drop('fold').drop('model')].mean().to_frame().T
                overall['model'] = i + '_overall_valid'
                subresults = pd.concat([subresults, overall])
                results = pd.concat([results, subresults])
                j += 1
            except:
                print('Not finished: ', path + i)
                continue
        results.sort_values(['model'],ascending=[False])
        results.to_csv('{}/valid/{}.csv'.format(task_result, model_name), index= False)
    except:
        continue