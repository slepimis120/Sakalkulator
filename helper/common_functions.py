
import csv
import os


def delete_files_in_folder(folder_path):
    try:
        files = os.listdir(folder_path)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if file_path == os.path.join(folder_path, "example.txt"):
                continue
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def load_csv(csv_path="data/audio/testing_data/res.csv"):
    data = {}
    csv_file = open(csv_path, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_file)

    next(csv_reader)  # preskoci zaglavlje
    for row in csv_reader:
        data[row[0]] = row[1]

    return data


def calculate_accuracy(results, csv_results):
    correct_predictions = 0
    total_predictions = 0
    for f, predicted in results.items():
        csv_result = csv_results[f[0:-4]]
        i = 0
        while i < min(len(csv_result), len(predicted)):
            if predicted[i] == csv_result[i]:
                correct_predictions += 1
            total_predictions += 1
            i += 1
        if len(csv_result) != len(predicted):
            total_predictions += abs(len(csv_result) - len(predicted))
    return round(correct_predictions/total_predictions, 4)


def print_calculation(result):
    try:
        print(result + " =", eval(result))
    except Exception as e:
        print(result)
