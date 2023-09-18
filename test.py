import torch
import argparse
import pandas as pd
import os

from train import initialize_model
from preprocess_data import create_custom_df, create_dataloaders, preprocess_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Add F1, RUC


def test_accuracy(data_loaders, model):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(preds.size())
            print(labels.size())

            num_correct += (preds == labels).sum()
            total += labels.size(0)

    print(f"Test Accuracy of the model: {float(num_correct) / float(total) * 100:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="model name")
    args = vars(parser.parse_args())
    batch_size = 64

    model_name = args["model"]

    # df = pd.read_pickle("Data/preprocessed_data.pkl")
    # num_classes = df["Classname"].nunique()

    dataset = "Custom"
    not_saved_custom = preprocess_images(dataset)
    df = create_custom_df(not_saved_custom)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("customData.csv")
    num_classes = df["Classname"].nunique()

    model, _ = initialize_model(model_name[:8], num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    model.load_state_dict(torch.load(path + "/models/" + str(model_name), map_location=device))
    model.to(device)
    model.eval()

    dataloaders = create_dataloaders(df, batch_size, "test")

    test_accuracy(dataloaders, model)


if __name__ == "__main__":
    main()
