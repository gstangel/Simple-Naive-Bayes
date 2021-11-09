from os.path import exists
from copy import copy


class NaiveBayes:
    def __init__(self):
        self.labels = list()  # list of the labels for the classes (ex: buying, maint, doors)
        self.classifications = dict()
        self.attributes = list()
        self.num_features = 0

    def load_metadata(self, filename):  # returns if the data was loaded successfully or not
        if exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
                for line in lines:  # iterate through all lines in the metadata
                    line = line.rstrip()
                    split_line = line.split(":")  # split the line
                    attribute = split_line[0]  # label for the class, for printing
                    possible_values = split_line[1]  # different possible classifications for the class
                    if attribute != "class":
                        self.attributes.append({classification: dict() for classification in possible_values.split(',')})
                        self.num_features += 1
                    else:
                        classifications = {classification: 1 for classification in possible_values.split(',')}
                        for attribute in self.attributes:
                            for key in attribute.keys():
                                attribute[key] = copy(classifications)
                        self.classifications = classifications
            print("***** Metadata successfully loaded. *****")
        else:
            print("ERROR: That file was not found")

    def train(self, filename):
        # get the counts
        if exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
                for datapoint in lines:
                    datapoint = datapoint.rstrip()
                    split_data = datapoint.split(',')
                    cur_class = split_data[-1]
                    # print(cur_class)
                    self.classifications[cur_class] += 1
                    for i in range(len(split_data) - 1): # iterate through all but the class
                        self.attributes[i][split_data[i]][cur_class] += 1
            # compute the probabilities of each attribute
            for attr in self.attributes: # list
                for key in attr.keys():
                    for class_ in self.classifications:
                        attr[key][class_] = attr[key][class_] / self.classifications[class_]
            
            # compute the probabilities of each class (total number of unacc, total number of acc, etc)
            total = 0
            for class_ in self.classifications:
                total += self.classifications[class_]
            for class_ in self.classifications:
                self.classifications[class_] = self.classifications[class_] / total
            print("***** Model successfully trained *****")
        else:
            print("ERROR: That file was not found")

    def test(self, filename, out_filename):
        if exists(filename):
            with open(filename, 'r') as file:
                with open(out_filename, 'w') as outfile:
                    num_correct = 0
                    lines = file.readlines()
                    for datapoint in lines:
                        datapoint = datapoint.rstrip()
                        split_data = datapoint.split(',')
                        predicted_class = None
                        max_probability = 0
                        for class_ in self.classifications:
                            prior_prob = self.classifications[class_]
                            for i in range(self.num_features):
                                prior_prob *= self.attributes[i][split_data[i]][class_]
                                
                            if prior_prob >= max_probability:
                                max_probability = prior_prob
                                predicted_class = class_
                        if len(split_data) > self.num_features:
                            if predicted_class == split_data[-1]:
                                num_correct += 1
                        outfile.write(str(datapoint + " / " + predicted_class + "\n"))
                if len(split_data) > self.num_features:
                    print(len(lines), "instances in the test data")
                    print(num_correct, "correctly classified")
                    print("Accuracy = " + str(num_correct) + "/" + str(len(lines)))
                else:
                    print(len(lines), "\ninstances in the test data")
        
                    print("No classifications were given for the test file, so we cannot report on accuracy")
        else:
            print("ERROR: That file was not found.")
            
    def debug(self):
        print(len(self.classifications))
        for attr in self.attributes:
            for key in attr.keys():
                print(key, " : ", attr[key])
            print("\n")
        for class_ in self.classifications:
            print(self.classifications[class_])
            
    def reset(self):
        self.labels = list() # list of the labels for the classes (ex: buying, maint, doors)
        self.classifications = dict()
        self.attributes = list()
        self.num_features = 0

    def mainmenu(self):
        print("***** Welcome to the Naive Bayes Classifier by Gabe St. Angel *****")
        while True:
            print("\n1. Load a Metadata File\n2. Load Training Data\n3. Run Testing Data\n4. Reset Classifier\n5. Exit")
            
            selection = input("Please enter a selction from the menu: ")
            if selection == "1":
                infile = input("Enter the name of the metadata file: ")
                self.load_metadata(infile)
            elif selection == "2":
                if self.classifications:
                    infile = input("Enter the name of the training data file: ")
                    self.train(infile)
                else:
                    print("ERROR: You must load the Metadata and training files first")
            elif selection == "3":
                if self.classifications:
                    infile = input("Enter the name of the test file: ")
                    outfile = input("Enter the name of the output file: ")
                    self.test(infile, outfile)
                else:
                    print("ERROR: You must load the Metadata and training files first")
            elif selection == "4":
                self.reset()
                print("Naive Bayes Classifier successfully reset")
            elif selection == "5":
                return
            else:
                print("ERROR: Invalid Selection")


if __name__ == "__main__":
    model = NaiveBayes()
    model.mainmenu()
