from src.dataset import load_ucihar

train, test = load_ucihar("data/UCI_HAR_Dataset")
print("Train X:", train.X.shape, "y:", train.y.shape, "subjects:", train.subjects.shape)
print("Test  X:", test.X.shape, "y:", test.y.shape, "subjects:", test.subjects.shape)
print("Labels:", train.label_map)