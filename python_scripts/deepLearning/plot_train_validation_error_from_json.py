import sys
import json
import matplotlib.pyplot as plt


def main():
    filename = sys.argv[1]
    
    with open(filename) as data_file:    
        data = json.load(data_file)
    print len(data)
    train_loss = [x["train_loss"] for x in data]
    epoch = [x["epoch"] for x in data]
    validation_loss = [x["val_error_rate"] for x in data]
    learning_rate = [x["learning_rate"] for x in data]
    plt.plot(epoch, train_loss, 'r', epoch, validation_loss, 'b', epoch, learning_rate, 'g')
    plt.show()
    

if __name__=="__main__":
    main()