text1 = open('Submission.csv', 'r')
text2 = open('submit_inception.csv', 'r')

text1 = text1.readlines()
text2 = text2.readlines()

label1 = []
label2 = []
for i in range(1, len(text1)):
    label1.append(text1[i].split(",")[1])
    label2.append(text2[i].split(",")[1])

# my_pred = text1.split(",")[0]
print(label1)
print(label2)

for i in range(len(text1)-1):
    if label2[i] == 'None\n':
        label2[i] = label1[i]

print(label2)

text3 = open("Submission_merged.csv", 'w')
text3.write("id;label\n")

for i in range(len(label2)):
    text3.write(str(i+1)+";"+label2[i])

