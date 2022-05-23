import pandas as pd


def load_data(path):
    """
        :parameter path: path to the data file
        :return: the pandas DataFrame
    """
    df = pd.read_csv(path)
    return df


def add_new_columns(df):
    #part 2:
    seasons = ["spring", "summer", "fall", "winter"]
    df["season_name"] = df["season"].apply(lambda x: seasons[x])

    ##part 3: לשנות מחר לחותמת זמן
    df["Hour"] = df["timestamp"].apply(lambda st: int(st[11:13]))
    df["Day"] = df["timestamp"].apply(lambda st: int(st[0:2]))
    df["Month"] = df["timestamp"].apply(lambda st: int(st[3:5]))
    df["Year"] = df["timestamp"].apply(lambda st: int(st[6:10]))

    # part 4:
    df["is_weekend_holiday"] = df.apply(lambda col: 2 * col["is_holiday"] + col["is_weekend"],axis=1)  # link to explanation (min 8): https://www.youtube.com/watch?v=6X6A8M2-Avg

    # part 5:
    df["t_diff"] = df.apply(lambda col: col["t2"] - col["t1"], axis=1)


def data_analysis(df):

    #part 6:
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()

    #part 7:
    temp_df = df.drop(["timestamp", "season_name"], axis=1)
    temp_dict = {}
    for first in temp_df.columns:
        for second in temp_df.columns:
            if temp_df.columns.get_loc(first) < temp_df.columns.get_loc(second):
                temp_tuple = (first, second)
                temp_dict[temp_tuple] = format(abs(corr[first][second]), ".6f")

    sorted_by_corr = sorted(temp_dict.items(), key=lambda t: t[1])
    print("Highest correlated are: ")
    for i in range(5):
        print(f"{i + 1}. {sorted_by_corr[-(i + 1)][0]} with {sorted_by_corr[-(i + 1)][1]}")
    print()
    print("Lowest correlated are: ")
    for i in range(5):
        print(f"{i + 1}. {sorted_by_corr[i][0]} with {sorted_by_corr[i][1]}")
    print()

    #part 8:
    seasons = ["fall", "spring", "summer", "winter"]
    temp = df.groupby('season_name').t_diff.mean()
    for season in seasons:
        print(f'{season} average t_diff is {format(temp[season], ".2f")}')
    print(f'All average t_diff is {format(df["t_diff"].mean(), ".2f")}')

















