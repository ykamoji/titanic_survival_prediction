import csv

datapath = '/Users/ykamoji/Documents/Semester1/COMPSCI_589/titanic_survival_prediction/titanic/'

headers = ["FamilyName", "FirstName"]

def load_data_web(filename):
    l = []
    with open(datapath + filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            first_name = row["FirstName"]
            family_name = row["FamilyName"]
            full_name = family_name.strip().lower() + ', ' + first_name.strip().lower()
            full_name = full_name.replace('.', '')
            l.append(full_name)

    return l

original_csv = []
original_csv_headers = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
new_csv_headers = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

def scrambled_name_check(person, search):
    partial_name = False
    # print(survived)
    names_split = search.replace(',', '').split(' ')
    # print(names_split)
    if len(names_split) > 0:
        if sum([1 if name in person else 0 for name in names_split]) == len(names_split):
            partial_name = True

    return partial_name

def update_survive_data(name, val):
    for person in original_csv:
        if name == person['Name']:
            person['Survived'] = val
            break
def process():

    survived_list = load_data_web('survivorList.csv')

    # print(survived_list)

    victims_list = load_data_web('victimsList.csv')

    # print(victims_list)

    # test_dict = load_data_test('test.csv')

    # print(test_dict)

    with open(datapath + 'test.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['Survived'] = -1
            original_csv.append(row)
            full_name = row["Name"]
            full_name = full_name.strip().lower()
            full_name = full_name.replace('.', '')

            for survived in survived_list:
                if (survived == full_name) or (survived in full_name) or scrambled_name_check(full_name, survived):
                    row['Survived'] = 1
                    break

            if row['Survived'] == 1: continue

            for victim in victims_list:
                # print(person +' =?= ' +survived)
                if (victim == full_name) or (victim in full_name) or scrambled_name_check(full_name, victim):
                    row['Survived'] = 0
                    break


    total = len(original_csv)

    found_survivors = sum([1 if person['Survived']==1 else 0 for person in original_csv])

    found_victims = sum([1 if person['Survived'] == 0 else 0 for person in original_csv])

    print(f"Size of test {total}")

    print(f"Found  {found_survivors} survivors")

    print(f"Found {found_victims} victims")

    print(f"Remaining {total - found_survivors - found_victims}")

    print("\n\n")


    # for item in original_csv:
    #     print(item)

    # with open('test_found.csv','w') as csvfile:
    #     writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=new_csv_headers, extrasaction='ignore')
    #     writer.writeheader()
    #     for data in original_csv:
    #         writer.writerow(data)


def process_extracted_data():
    extrated_data = []
    with open(datapath + 'test_found.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            extrated_data.append(row)

    total = len(extrated_data)

    found_survivors = sum([1 if person['Survived'] == '1' else 0 for person in extrated_data])

    found_victims = sum([1 if person['Survived'] == '0' else 0 for person in extrated_data])

    print(f"Size of test {total}")

    print(f"Found  {found_survivors} survivors")

    print(f"Found {found_victims} victims")

    print(f"Remaining {total - found_survivors - found_victims}")


if __name__ == '__main__':
    pass
    # process()
    # process_extracted_data()