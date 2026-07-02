import pickle, json
import os.path
from sklearn import model_selection as cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from time import time
from sklearn.svm import SVC
from room_type import RoomType

class SVCRoomClassifier:
    ###
    # Initialises the SVC- either by training from the training data or by loading it from
    # a pickle file- depending on the TRAIN_FROM_SCRATCH parameter.
    ###
    def __init__(self, TRAIN_FROM_SCRATCH = False):

        self.load_training_data_and_vectorize()

        # We can either load pre-trained settings or train from scratch
        if TRAIN_FROM_SCRATCH:
            # We will use an Support Vector Classifier
            self.clf = SVC(kernel="rbf", C=10000.0)

            # Now let's train our SVC
            t0 = time()
            self.clf.fit(self.features_train_vectorized, self.labels_train)
            #print("training time:", round(time()-t0, 3), "s")

            # Now let's save our trained parameters
            pickle.dump(self.clf, open("trained_svc_room_classifier.pkl", "wb"))
        else: # so we want to load pre-trained parameters
            with open("trained_svc_room_classifier.pkl", "rb") as file:
                self.clf = pickle.load(file)

        t0 = time()
        self.pred = self.clf.predict(self.features_test_vectorized)
        #print("prediction time:", round(time()-t0, 3), "s")

        # Finally learn and test how good model have we got
        t0 = time()
        self.acc = accuracy_score(self.pred, self.labels_test)
        #print("accuracy calculation time:", round(time()-t0, 3), "s")

        print("Accuracy = ",self.acc)
        #print(self.clf.classes_)
        #########################################################

    def getAccuracy(self):
        return self.acc

    ###
    # Loads training data from pickle files and vecorizes the data for use in the SVC
    ###
    def load_training_data_and_vectorize(self):
        with open('data.json', 'r') as f:
            # json.dump(data_set, f)
            training_data = json.load(f)
            labels = [label for (label, features) in training_data]
            features = [features for (label, features) in training_data]

        # Now create training and testing sets
        features_train, features_test, self.labels_train, self.labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=1983)

        # Now we will turn the texts into numerical vectors so that we can use that for machine learning
        #vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
        self.vectorizer = TfidfVectorizer()

        # If we want to give all features and all labels to training (leaving no unique test cases), then uncomment below
        #features_train = features
        #labels_train = labels

        self.features_train_vectorized = self.vectorizer.fit_transform(features_train)
        self.features_test_vectorized  = self.vectorizer.transform(features_test)

        #print "tfidf.get_stop_words(): ",tfidf.get_stop_words()
        #print "vector: ",vector
        #print(self.features_train_vectorized.shape)
        #print(self.features_test_vectorized.shape)
        #print(self.vectorizer.get_feature_names_out())

    def classify_room_by_this_object_set(self, obj_set):
        # now we'll get the objects into a string separated by a space
        objs_in_room_as_string = ""
        for obj in obj_set:
            objs_in_room_as_string += obj + " "

        objs_in_room_as_string = objs_in_room_as_string[:-1]

        ans = self.predict(objs_in_room_as_string)

        #print("\n" + str(ans) + " :: " + list(self.room_types.keys())[list(self.room_types.values()).index(ans)])
        #print("\n" + ans.name + " :: " + str(ans.value))

        return ans

    ###
    # Uses the cassifier to predict a room based on the input elements found in the room
    ###
    def predict(self, items_as_string_separated_by_space):
        input_vectorized  = self.vectorizer.transform([items_as_string_separated_by_space])
        #print(input_vectorized)

        #t0 = time()
        result = self.clf.predict(input_vectorized)
        #print("svc predict time:", round(time()-t0, 5), "s")

        print("Prediction of: " + items_as_string_separated_by_space + " : " + result[0])

        return RoomType.interpret_label(result[0])

def main():
    rc = SVCRoomClassifier(True)
    rc.predict("WATCH DRESSER DININGTABLE BOOTS")

if __name__ == "__main__":
    main()
