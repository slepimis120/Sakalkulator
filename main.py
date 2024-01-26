import sys


def read_data():
    data_type = sys.argv[1]
    if data_type == "video":
        print("Video")
    elif data_type == "audio":
        pass
    else:
        print("Wrong data type!")


if __name__ == '__main__':
    read_data()
