from flair_project.config import registry
from flair.nn import Classifier


@registry.architectures("flair.ner.v1")
def build_ner():
    return Classifier.load("ner")
