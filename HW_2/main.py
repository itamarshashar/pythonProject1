from clustering import *
from data import *


def main():
    # part A
    df_A = load_data("C:\data_science\HW_2\london_sample_500.csv")
    print("Part A:")
    add_new_columns(df_A)
    data_analysis(df_A)
    print()


    # part B
    df_B = load_data("C:\data_science\HW_2\london_sample_500.csv")
    print("Part B:")
    temp = transform_data(df_B, ["cnt", "hum"])


    #tests

    

    print('centroids for k = 2')
    labels, centroids = kmeans(temp, 2)
    print(np.array_str(centroids, precision=3, suppress_small=True))
    visualize_results(temp, labels, centroids,"C:\data_science\HW_2\plots\plot{}.png")
    print()
    

    print('centroids for k = 3')
    labels, centroids = kmeans(temp, 3)
    print(np.array_str(centroids, precision=3, suppress_small=True))
    print()
    visualize_results(temp, labels, centroids,"C:\data_science\HW_2\plots\plot{}.png")

    print('centroids for k = 5')
    labels, centroids = kmeans(temp, 5)
    print(np.array_str(centroids, precision=3, suppress_small=True))
    visualize_results(temp, labels, centroids,"C:\data_science\HW_2\plots\plot{}.png")



if __name__ == "__main__":
    main()








