def clean(data):
    data = data.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col] = data[col].fillna(data[col].median())
    data["Embarked"] = data["Embarked"].fillna("U")
    return data
    
import pandas as pd
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_id = test["PassengerId"]
data = clean(data)
test = clean(test)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cols = ["Sex", "Embarked"]
for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x = data.drop("Survived", axis=1)
y = data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=42, max_iter=1000).fit(x_train, y_train)
print(clf.score(x_test, y_test))
submission = clf.predict(test)
df = pd.DataFrame({"PassengerId": test_id, "Survived": submission})
df.to_csv("submission.csv", index=False)