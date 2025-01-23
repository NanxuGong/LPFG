from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors
from sklearn.linear_model import Lasso, Ridge, LogisticRegression # 
from xgboost import XGBClassifier # XGBoost
from lightgbm import LGBMClassifier # LightGBM
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from transformers import pipeline
import os
import openai
from sklearn.naive_bayes import GaussianNB 
import torch
import data
from sklearn.metrics import accuracy_score
from tabpfn.scripts import tabular_metrics
from functools import partial
from caafe.preprocessing import make_datasets_numeric
import pandas as pd
# from logger import *
from Operation import *
# from tools import *
import re
import warnings
from lightgbm import LGBMClassifier # LightGBM
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import time
from autofeat import AutoFeatRegressor, AutoFeatModel
from autofeat import AutoFeatClassifier
openai.api_key = "Your key"

np.random.seed(30)
# torch.manual_seed(42)
def test(train_x, train_y, test_x, test_y):
    print(train_x.shape)
    clf = RandomForestClassifier()
    # clf = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    # clf.fit = partial(clf.fit, overwrite_warning=True)

    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    acc = accuracy_score(pred, test_y)
    return acc

def generator_prompt(df_train, target_column_name, dataset_description, advice, column_name, train_x):
    sys = "You are a data scientist. Given the task description and the dataset, you are generating new features by combining original features. "
    prompt = "Task prompt: " + dataset_description
    prompt += "Dataset: "
    for i, name in enumerate(column_name):
        feature = train_x[:, i]
        mean = torch.mean(feature).item()
        std = torch.std(feature).item()
        min_val = torch.min(feature).item()
        max_val = torch.max(feature).item()
        prompt += f"Feature: {name}, mean: {mean:.2f}, std: {std:.2f}, min: {min_val:.2f}, max: {max_val:.2f}"
    prompt += "Operator set: sqrt, square, sin, cos, tanh, stand_scaler, minmax_scaler, quan_trans, sigmoid, log, reciprocal, cube, +, -, *, / "
    prompt += """Let’s think step by step.
    Step 1. Understand the task and think about how to generate new features according to my advice. Note that you can only generate features using the given operators and features.
    Step 2. Formatting the generated features as a sequence consisting of feature IDs and operators. Everytime generate no more than 2 features. Please ensure the the feature sequence and generated feature are accurate and consistent
    Format for Response:
    - Feature Sequence: …
    - Generated Features: …

    Each feature is represented by an ID (e.g., f0, f1, f2) based on its position in the dataset. Do not include the original feature names in the generated sequences. A valid example of a feature sequence is:
    [ ( ( f0 * f2 ) + f6 ), ( sqrt f1 ), ( sin f4 ), ( f3 / f5 ) ]
    Ensure that:
	1.	The entire feature sequence is enclosed within square brackets [ ].
	2.	Each token is separated by a single space.

    Format for generated feature

    Give the mathematical operation for each generated feature. Note the feature name should be correct.
    e.g., New feature 1 = chlorides * fixed acidity

    Here is my advice: 
    """
    prompt += advice

    return {"role": "system", "content": sys}, {"role": "user", "content": prompt}

def optimizer_prompt(df_train, target_column_name, dataset_description, column_name, train_x):
    sys = "You are a data scientist. Given the task description and the dataset, you are advising on how to generate new features by combining original features. "
    prompt = "Task prompt: " + dataset_description
    prompt += "Dataset: "
    for i, name in enumerate(column_name):
        feature = train_x[:, i]
        mean = torch.mean(feature).item()
        std = torch.std(feature).item()
        min_val = torch.min(feature).item()
        max_val = torch.max(feature).item()
        prompt += f"Feature: {name}, mean: {mean:.2f}, std: {std:.2f}, min: {min_val:.2f}, max: {max_val:.2f}"
    prompt += """
    Let’s think step by step.
    Step 1. Analyze the semantics of features and task. Advise on generating semantically informative features.
    Step 2. Analyze the distribution of features. Advise on how to generate features to improve the data distribution.
    Note that your advice should be short, general, and no examples. Everytime give a piece of advice from these perspectives.
    Format for Response:

    - Advice on semantics: …
    - Advice on data: … 
    """
    return {"role": "system", "content": sys}, {"role": "user", "content": prompt}

def generator_next(advice):
    prompt = "Here is my new advice, continue to generate new features: " + advice
    return prompt

def optimizer_next(features):
    prompt = "Here is the generated features, give me more advice from different perspective: " + features
    return prompt
# clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)

def caafe(df_train, df_test,target_column_name, dataset_description, test_y):
    clf_no_feat_eng = RandomForestClassifier(random_state=22)
    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                llm_model="gpt-3.5-turbo",
                                iterations=2)

    caafe_clf.fit_pandas(df_train,
                        target_column_name=target_column_name,
                        dataset_description=dataset_description)

    pred = caafe_clf.predict(df_test)
    acc = accuracy_score(pred, test_y)
    print(f'Accuracy after CAAFE {acc}')


def evaluation(feature_seq, Dg_train, Dg_test, train_y, test_y, iteration):
    if len(feature_seq) == 0:
        print("no generated features")
        sys.exit()
    else:
        for seq in feature_seq:
            trans = seq.replace('f', '')
            trans = trans.split(',')
        for i in trans:
            try:
                ops = show_ops_r(converge(i.split()))
                print(ops)
                Dg_test[i] = op_post_seq(Dg_test, ops)
                Dg_train[i] = op_post_seq(Dg_train, ops)
            except:
                continue

    new_train_x = torch.tensor(Dg_train.values)
    new_test_x = torch.tensor(Dg_test.values)
    acc = test(new_train_x, train_y, new_test_x, test_y)
    print(f'Accuracy after itration {iteration}: {acc}')
    
    return Dg_train, Dg_test

def baseline():
    metric_used = tabular_metrics.auc_metric
    cc_test_datasets_multiclass = data.load_all_data()
    
    print(len(cc_test_datasets_multiclass))
    ds = cc_test_datasets_multiclass[2]
    ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
    column_name = ds[4][:-1]
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]

    df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
    # del cc_test_datasets_multiclass[6]
    # del cc_test_datasets_multiclass[1]   
    # del cc_test_datasets_multiclass[0]
    # del cc_test_datasets_multiclass[0]
    # del cc_test_datasets_multiclass[0]
    # del cc_test_datasets_multiclass[0]
    # del cc_test_datasets_multiclass[0]
    # del cc_test_datasets_multiclass[0]
    # for ds_ in cc_test_datasets_multiclass:
    ds_ = cc_test_datasets_multiclass[0]
    ds, df_train, df_test, _, _ = data.get_data_split(ds_, seed=0)
    column_name = ds[4][:-1]
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]
    time0 = time.time()
    df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
    train_x, train_y = data.get_X_y(df_train, target_column_name)
    test_x, test_y = data.get_X_y(df_test, target_column_name)

    acc = test(train_x, train_y, test_x, test_y)
    print(f'Accuracy {ds[0]} FG {acc}')
    
    caafe(df_train, df_test, target_column_name, dataset_description, test_y)
    time1 = time.time()
    print(time1 - time0)
    return 0

def main():
    now_iteration = 0
    metric_used = tabular_metrics.auc_metric
    cc_test_datasets_multiclass = data.load_all_data()
    
    print(len(cc_test_datasets_multiclass))
    
    # del cc_test_datasets_multiclass[6]
    # del cc_test_datasets_multiclass[1]  
    ds = cc_test_datasets_multiclass[args.task_id]
    ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
    column_name = ds[4][:-1]
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]
    print(column_name)
    print(target_column_name)
    print(dataset_description)
    print('-----------------Performing feature generation on dattaset: ', ds[0], '-----------------')
    df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
      

    # baseline(cc_test_datasets_multiclass)  
    train_x, train_y = data.get_X_y(df_train, target_column_name)
    test_x, test_y = data.get_X_y(df_test, target_column_name)
    
    acc_before = test(train_x, train_y, test_x, test_y)
    print(f'Accuracy before FG {acc_before}')
    # caafe(df_train, df_test)

    time0 = time.time()
    sys_optimizer, prt_optimizer = optimizer_prompt(train_x, target_column_name, dataset_description, column_name, train_x) # "what's your advice"#
    # print(prt_optimizer)
    mssa_optimizer = [sys_optimizer, prt_optimizer]
    optimizer_model = openai.ChatCompletion.create(
        model=args.gpt_model,  # 或者使用其他可用模型，例如 "gpt-4-turbo"
        messages=mssa_optimizer,
        # max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.7,
    )

    generated_optimizer = optimizer_model['choices'][0]['message']['content'] #"advice 1, advice 2"#
    print(generated_optimizer)
    sys_generator, prt_generator = generator_prompt(df_train, target_column_name, dataset_description, generated_optimizer, column_name, train_x) #"what are the generated features"#
    # print(prt_generator)
    mssa_generator = [sys_generator, prt_generator]
    generator_model = openai.ChatCompletion.create(
        model=args.gpt_model,  # 或者使用其他可用模型，例如 "gpt-4-turbo"
        messages=mssa_generator,
        # max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.7,
    )

    generated_generator = generator_model['choices'][0]['message']['content'] #"generated eature are"#
    print(generated_generator)
    mssa_generator.append({'role': 'assistant', 'content': generated_generator})
    feature_seq = re.findall(r'\[(.*?)\]', generated_generator) # feature_seq = ( f0 + f1 * f2 ), ( f3 - f4 ), ( f5 * f6 + f7 ), ( f8 / f9 ) ]'#
    print(feature_seq)
    
    train_x_np = train_x.numpy()
    test_x_np = test_x.numpy()
    # 创建 DataFrame

    Dg_train = pd.DataFrame(train_x_np, columns=column_name)
    Dg_test = pd.DataFrame(test_x_np, columns=column_name)
    Dg_train, Dg_test = evaluation(feature_seq, Dg_train, Dg_test, train_y, test_y, now_iteration)

    now_iteration += 1

    while now_iteration < args.iteration:
        mssa_optimizer.append({'role': 'assistant', 'content': generated_optimizer})
        start_index = generated_generator.find("Generated Features")
        if start_index != -1:
            extracted_text = generated_generator[start_index:] #'new feature 1: ' #
            # print(extracted_text)
            mssa_optimizer.append({'role': 'user', 'content': optimizer_next(extracted_text)})
            optimizer_model = openai.ChatCompletion.create(
            model=args.gpt_model,  # 或者使用其他可用模型，例如 "gpt-4-turbo"
            messages=[
                sys_optimizer, prt_optimizer
            ],
            # max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.7,
            )
            generated_optimizer = optimizer_model['choices'][0]['message']['content'] #"advice 1, advice 2"#
            # print(generated_optimizer)
            # sys_generator, prt_generator = generator_prompt(df_train, target_column_name, dataset_description, generated_optimizer, column_name, train_x) #"what are the generated features"#
            mssa_generator.append({'role': 'user', 'content': generator_next(generated_optimizer)})
            generator_model = openai.ChatCompletion.create(
                model=args.gpt_model,  # 或者使用其他可用模型，例如 "gpt-4-turbo"
                messages=mssa_generator,
                # max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.7,
            )
            generated_generator = generator_model['choices'][0]['message']['content'] #"generated eature are"#
            mssa_generator.append({'role': 'assistant', 'content': generated_generator})
            # print(generated_generator)
            time1 = time.time()
            feature_seq = re.findall(r'\[(.*?)\]', generated_generator)
            print(feature_seq) # feature_seq = ( f0 + f1 * f2 ), ( f3 - f4 ), ( f5 * f6 + f7 ), ( f8 / f9 ) ]'#
            Dg_train, Dg_test = evaluation(feature_seq, Dg_train, Dg_test, train_y, test_y, now_iteration)
            print(f'Iteration {now_iteration} time: {time1 - time0}')
            now_iteration += 1
        else:
            print(generated_generator)
            print("Can not find 'Generated Features'")
            sys.exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Generation Script')
    parser.add_argument('--iteration', type=int, default=1, help='Number of iterations for feature generation')
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help='Choice of GPT model')
    parser.add_argument('--task_id', type=int, default=0, help='Dataset ID')
    args = parser.parse_args()
    # main()
    # baseline()
    main()  
    