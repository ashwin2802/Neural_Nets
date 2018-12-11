import os

num = 0
def run():
    
    cmd = raw_input("Enter command: ")

    if(cmd=="build"):
        if(num==1):
            bin = raw_input("Model has already been built. Do you wish to train it again? (y/n): ")
            if(bin=="y"):
                print("Deleting existing model.")
                os.system('rm -rf ./model/model.json')
                os.system('rm -rf ./model/model.h5')
                os.system('python ./scripts/build.py')
                num = 1
                run()
            elif(bin=="n"):
                print("Select a different command: evaluate, predict, check, exit.")
                run()
        elif(num==0):
            os.system('python ./scripts/build.py')
            num = 1
            run()

    elif(cmd=="check"):
        os.system('python ./scripts/check.py')
        run()

    elif(cmd=="predict"):
        if(num==0):
            print("Model not found, building a new one.")
            os.system('python ./scripts/build.py')
            num = 1
        os.system('python ./scripts/predict.py')
        run()

    elif(cmd=="evaluate"):
        if(num==0):
            print("Model not found, building a new one.")
            os.system('python ./scripts/build.py')
            num = 1
        os.system('python ./scripts/eval.py')
        run()

    elif(cmd=="exit"):
        exit()

    else:
        print("Please enter a valid command. Available commands: build, evaluate, predict, check, exit.")
        run()


print("List of available commands: build, evaluate, predict, check, exit.")
run()

