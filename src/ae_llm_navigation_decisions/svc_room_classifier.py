import pickle, json
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from importlib.resources import files
from .room_type import RoomType

class SVCRoomClassifier:
    def __init__(self, train_from_scratch=False, model_path=''):

        if model_path == '':
            model_path = files('ae_llm_navigation_decisions.models').joinpath('room_classifier.pkl')

        self.model_path = model_path

        if train_from_scratch:
            self._train()
            self._save_model()
        else:
            self._load_model()

        # Evaluate on test set
        self._evaluate()

    def _train(self):
        """Train the classifier from scratch"""
        # Load and prepare data
        features_train, features_test, labels_train, labels_test = self._load_and_split_data()

        # Vectorize
        self.vectorizer = CountVectorizer(binary=True)  # Presence/absence only
        features_train_vec = self.vectorizer.fit_transform(features_train)
        features_test_vec = self.vectorizer.transform(features_test)

        # Cross-validation to find best parameters
        print("Performing cross-validation...")
        param_grid = {'C': [0.1, 0.5, 1.0, 5.0, 10.0], 'kernel': ['rbf', 'linear']}
        grid_search = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(features_train_vec, labels_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        # Train final model
        self.clf = grid_search.best_estimator_
        self.clf.fit(features_train_vec, labels_train)

        # Store test data for evaluation
        self.features_test_vec = features_test_vec
        self.labels_test = labels_test

    def _load_and_split_data(self):
        """Load data and split into train/test sets"""
        json_path = files('ae_llm_navigation_decisions.models').joinpath('data.json')
        with open(json_path, 'r') as f:
            training_data = json.load(f)
            labels = [label for (label, features) in training_data]
            features = [features for (label, features) in training_data]

        # Convert features from lists to strings
        features_str = [' '.join(f) for f in features]
        print(features_str)
        return model_selection.train_test_split(
            features_str, labels, test_size=0.2, random_state=42, stratify=labels
        )

    def _evaluate(self):
        """Evaluate model on test set"""
        if hasattr(self, 'features_test_vec') and hasattr(self, 'labels_test'):
            predictions = self.clf.predict(self.features_test_vec)
            accuracy = accuracy_score(self.labels_test, predictions)
            print(f"\nTest Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(self.labels_test, predictions))

    def _save_model(self):
        """Save model and vectorizer"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.clf,
                'vectorizer': self.vectorizer,
                'classes': self.clf.classes_
            }, f)
        print(f"Model saved to {self.model_path}")

    def _load_model(self):
        """Load model and vectorizer"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.clf = data['classifier']
            self.vectorizer = data['vectorizer']
        print(f"Model loaded from {self.model_path}")

    def predict(self, objects):
        """
        Predict room type from a list of objects.

        Args:
            objects: List of object names or space-separated string

        Returns:
            RoomType enum value
        """
        if isinstance(objects, list) or isinstance(objects, set):
            objects_str = ' '.join(objects)
        else:
            objects_str = objects

        vectorized = self.vectorizer.transform([objects_str])
        result = self.clf.predict(vectorized)[0]

        print(f"Predicted: {result} from objects: {objects_str}")
        return RoomType.interpret_label(result)

    def predict_proba(self, objects):
        """Get probability scores for each room type"""
        if isinstance(objects, list) or isinstance(objects, set):
            objects_str = ' '.join(objects)
        else:
            objects_str = objects

        vectorized = self.vectorizer.transform([objects_str])
        probabilities = self.clf.predict_proba(vectorized)[0]

        return dict(zip(self.clf.classes_, probabilities))


def main():
    # Train from scratch
    classifier = SVCRoomClassifier(train_from_scratch=False, model_path="models/room_classifier.pkl")

    # Test prediction
    result = classifier.predict(['WATCH', 'DRESSER', 'DININGTABLE', 'BOOTS'])
    print(f"Room type: {result}")
    print(classifier.predict_proba(['WATCH', 'DRESSER', 'DININGTABLE', 'BOOTS']))

    result = classifier.predict(['TOWELHOLDER', 'TOILET', 'TOWEL', 'TOILETPAPER', 'CLOTHESDRYER'])
    print(f"Room type: {result}")
    print(classifier.predict_proba(['TOWELHOLDER', 'TOILET', 'TOWEL', 'TOILETPAPER', 'CLOTHESDRYER']))

    print(classifier.predict_proba(['CHAIR', 'WINDOW']))


if __name__ == "__main__":
    main()