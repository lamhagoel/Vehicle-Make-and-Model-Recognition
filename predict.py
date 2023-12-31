import argparse
import torch
from torchvision import transforms
import pandas as pd
import os

from train import initialize_model

from preprocess_data import preprocess_images, create_custom_df


from PIL import Image

device = torch.device("cpu")


def predict(file, model_name="resnet50_100epochs.pt", k=5):

    img_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = Image.open(file)
    if img.mode != "RGB":  # Convert png to jpg
        img = img.convert("RGB")
    img = img_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # Add batch dimension (because single image)
    print(img.size())

    # df = pd.read_pickle("Data/preprocessed_data.pkl")
    # num_classes = df["Classname"].nunique()
    # num_classes = 100

    dataset = "Custom"
    not_saved_custom = preprocess_images(dataset)

    # df_stanford = create_stanford_df(not_saved_stanford)
    # df_vmmrdb = create_vmmrdb_df(not_saved_vmmr, min_examples=100)
    # df, num_classes = create_unified_df(df_stanford, df_vmmrdb)
    # print(f'df created!')
    # print(f'num_classes: {num_classes}')
    df = create_custom_df(not_saved_custom)
    df.reset_index(drop=True, inplace=True)
    df["Classencoded"] = df["Classname"].factorize()[0]
    print(df.info())
    num_classes = df["Classname"].nunique()
    # print(df["Classname"].unique())

    model, _ = initialize_model(model_name[:8], num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    model.load_state_dict(torch.load(path + "/models/" + str(model_name), map_location=device))
    model.to(device)
    model.eval()

    pd.set_option('display.max_rows', None)

    with torch.no_grad():
        output = model(img)
        print(output.shape)
        output = torch.nn.functional.softmax(output, dim=1)
        probs, preds = torch.topk(output, k)

    preds = torch.transpose(preds, 0, 1)
    preds = preds.cpu()  # Send tensor to cpu
    preds = pd.DataFrame(preds.numpy(), columns=["Classencoded"])  # Convert to dataframe

    probs = torch.transpose(probs, 0, 1)
    probs = probs.cpu()

    class_encoded_matches = pd.merge(df, preds, how="inner")
    class_encoded_matches = pd.merge(preds, class_encoded_matches, how="left", on="Classencoded", sort=False)  # Preserves ordering
    classname_matches = class_encoded_matches["Classname"].unique()

    return classname_matches, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to image")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-k", "--topk", type=int, help="top k predictions")
    args = vars(parser.parse_args())

    path = args["path"]
    model_name = args["model"]
    k = args["topk"]

    classname_matches, probs = predict(path, model_name, k)
    print(classname_matches)
    print(probs)


if __name__ == "__main__":
    main()
