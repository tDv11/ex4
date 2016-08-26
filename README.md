# ex4<br>
<br>
Crowdflower Search Results Relevance .<br>
<br>
The preprocessing was all about word replacement, spelling correction and synonym replacement for deep learning.<br>
we started with building a model using ensemble selection, by using a library that is intended to be as diverse as possible to<br>
capitalize on a large number of unique learning approaches.<br>
to decode we calculated the precentage of any relevant level, by that we rank the prediction in order and gave weight for the models.<br>
```python
train = pickle.load(open('train_extracted_df.pkl', 'rb'))
test = pickle.load(open('test_extracted_df.pkl', 'rb'))
y_train = train["median_relevance"]

features = ['query_tokens_in_title', 'query_tokens_in_description', 'percent_query_tokens_in_description', 'percent_query_tokens_in_title', 'query_length', 'description_length', 'title_length', 'two_grams_in_q_and_t', 'two_grams_in_q_and_d']

#Random forest
print("random_forest_model")
model = RandomForestClassifier(n_estimators=300, n_jobs=1, min_samples_split=10, random_state=1, class_weight='auto')
rf_final_predictions, rf_score = ouput_final_model(model, train, test, features)
pickle.dump(rf_final_predictions, open('rf_final_predictions.pkl', 'wb'))

#SVC
print("SVC_model")
scl = StandardScaler()
svm_model = SVC(C=10.0, random_state = 1, class_weight = {1:2, 2:1.5, 3:1, 4:1})
model = Pipeline([('scl', scl), ('svm', svm_model)])
svc_final_predictions, svc_score = ouput_final_model(model, train, test, features)
pickle.dump(svc_final_predictions, open('svc_final_predictions.pkl', 'wb'))

```
<br>
In order to improve the similarity between the searches and their description we wanted to see all the possible combinations tokens of sting and then compare them.
```python
def get_n_string_similarity(s1, s2, n):
    str1 = set(get_n(str1, n))
    str2 = set(get_n(str2, n))
    if len(s1.union(str2)) == 0:
        return 0
    else:
        return float(len(str1.intersection(str2)))/float(len(str1.union(str2)))

def get_n(str, n):
    token_pattern = re.compile(r"(?u)\b\w+\b")
    word_list = token_pattern.findall(str)
    ng = []


    if n > len(word_list):
        return []
    
    for i, word in enumerate(word_list):
        ng = word_list[i:i+n]
        if len(ng) == n:
            n.append(tuple(ng))
    return ng
```
we mostly relied on numpy and SVD, and for the training part, on keras.<br>
<br>
The ScreenShot:<br>
![alt tag](SSPosition.jpg)
