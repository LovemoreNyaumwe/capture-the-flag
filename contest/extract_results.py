
if __name__ == '__main__':
    f = open("resultsBlue.txt", "r")
    counter = 0
    lines = f.readlines()
    results = []
    for line in lines:
        line = line.strip()
        if any(char.isdigit() for char in line) and len(line) > 1:
            for x in line.split(" "):
                if x.isdigit():
                    results.append(int(x))
            counter += 1
            print line
    print counter
    print float(sum(results))/100