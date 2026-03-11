import pandas as pd

input_csv = "/project2/shrikann_35/hongn/SpeechGesSyn-CARE/clips.csv"
output_csv = "/project2/shrikann_35/hongn/SpeechGesSyn-CARE/clips2.csv"

old_prefix = "/project2/shrikann_35/hongn/"
new_prefix = "/project2/shrikann_35/DATA/"

df = pd.read_csv(input_csv)

df["path_to_clip"] = df["path_to_clip"].str.replace(
    old_prefix,
    new_prefix,
    regex=False
)

df.to_csv(output_csv, index=False)
