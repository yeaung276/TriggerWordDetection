from Utils.td_utils import load_raw_audio

activates, negatives, backgrounds = load_raw_audio()

print("background length: " + str(len(backgrounds[0])),"\n")
print("Number of background: " + str(len(backgrounds)),"\n")
print("Number of activate examples: " + str(len(activates)),"\n")
print("Number of negative examples: " + str(len(negatives)),"\n")