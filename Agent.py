# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np
import cv2


class Node:
    def __init__(self, picture):
        self.picture = picture
        self.img2 = Image.open(picture).convert('L')         # convert image to greyascale
        self.img = Image.open(picture)
        self.imgArray = np.asarray(self.img)
        self.inverted = np.asarray(self.img)
        self.inverted = np.where(self.inverted < 127.5, 255, 0)
        self.cvimg = cv2.imread(picture)
        self.gray = cv2.cvtColor(self.cvimg, cv2.COLOR_BGR2GRAY)
        #convert to binary image where each pixel is eithr black or white
        _, self.binary = cv2.threshold(self.gray, 127, 1, cv2.THRESH_BINARY_INV)
        self.blackRatio = self.blackRatioCalc(self.binary)
        self.shapes = set()


    def blackRatioCalc(self, arr):
        return (np.sum(arr)/(184*184))

    def createNode(self, array):
        self.imgArray = array


class Frame:
    def __init__(self):

        self.nodes = dict()
        self.PROBABILITY = (184*184)*255*0.06 #1% of maxValye of array

    def createFrame(self, problem):
        for item in  problem.keys():
            self.nodes[item] = Node(problem[item].visualFilename)


    def createHorizonatalRefNode(self, node):
        tempNode1 = Node(node.picture)
        tempNode1.imgArray = np.flip(node.imgArray, 1)
        return tempNode1

    def createVerticalRefNode(self, node):
        tempNode1 = Node(node.picture)
        tempNode1.imgArray = np.flip(node.imgArray, 0)
        return tempNode1


    def compare(self, node1, node2):
        return np.sum(node1.imgArray-node2.imgArray) ==0

    def probablisticCompare(self, node1, node2):
        return abs(np.sum(node1.imgArray - node2.imgArray)) <= self.PROBABILITY

    def probablisticCompareBinary(self, node1, node2):
        return abs(np.sum(node1.binary - node2.binary)) <= (184*184)*255*0.1 

    def variableProbablisticCompare(self, node1, node2, probability):
        return abs(np.sum(node1.imgArray - node2.imgArray)) <= probability



    # return a set of of images that are all identical (not the options)
    def checkIdentical(self):
        identical = set()
        # print(self.nodes['A'])
        if self.compare(self.nodes['A'],self.nodes['B']):
            identical.add('A')
            identical.add('B')
        if self.compare(self.nodes['A'],self.nodes['C']) :
            identical.add('A')
            identical.add('C')
        if self.compare(self.nodes['C'],self.nodes['B']) :
            identical.add('C')
            identical.add('B')
        return identical


    def checkHorizontalReflection(self, node1, node2):
        tempNode1 = Node(node1.picture)
        tempNode1.imgArray = np.flip(node1.imgArray, 1)
        img3 = node1.imgArray-node2.imgArray
        if self.probablisticCompare(tempNode1, node2):
            return True
        return False


    def checkVerticalReflection(self, node1, node2):

        tempNode1 = Node(node1.picture)
        tempNode1.imgArray = np.flip(node1.imgArray, 0)
        img3 = node1.imgArray-node2.imgArray
        if self.probablisticCompare(tempNode1, node2):
            return True
        return False

    def findSimilar(self, node, similarity):
        # findSimilar takes in a node and finds a one similar to it from the options available
        for item in self.nodes.keys():
            if item not in {'A','B','C','D','E','F','G','H'}:
                if similarity == 1:
                    if self.compare(node,self.nodes[item]):
                        return int(item)
                elif similarity == 2:
                    if self.variableProbablisticCompare(node, self.nodes[item], (184*184)*255*0.023):
                        return int(item)
                elif similarity == 3:
                    if self.probablisticCompareBinary(node, self.nodes[item]):
                        return int(item)
                elif similarity == 4:
                    return self.mostSimilar(node)
                else:
                    if self.probablisticCompare(node,self.nodes[item]):
                        return int(item)

        return -1

    def mostSimilar(self, node1):
        similar = dict()
        for item in self.nodes.keys():
            if item not in {'A','B','C','D','E','F','G','H'}:
                similar[abs(np.sum(node1.imgArray - self.nodes[item].imgArray))] = item
        return similar[min(similar.keys())]



    def blackDifference(self, node1, node2):
        backlevel1 = np.sum(node1.binary)
        backlevel2 = np.sum(node1.binary)
        blackDifference = (blacklevel2-blacklevel1)/blacklevel1
        return blackDifference


    def addImage(self, node1, node2):
        temp = node1.inverted + node2.inverted
        temp = np.where(temp < 127.5, 255, 0)
        return temp
        
    def subtractImage(self, node1, node2):
        temp = node1.inverted - node2.inverted
        temp = np.where(temp < 127.5, 255, 0)
        return temp



class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        self.test = 1

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):

        frame = Frame()
        frame.createFrame(problem.figures)
        if problem.problemType == '2x2':

            identical = frame.checkIdentical()
            if len(identical)!=0:
                # if indentical has 3 elements within it, all 3 images are the same, thus the 4th image must also be the same
                if (len(identical)==3):
                    return frame.findSimilar(frame.nodes['A'], 1)
                else:
                # if identical has 2 elements within it, then we use the third one to find a match
                # note: we cannot have a set with one element in it, as it cannot be identical to itself
                    if 'A' not in identical:
                        return frame.findSimilar(frame.nodes['A'], 0)
                    elif 'B' not in identical:
                        return frame.findSimilar(frame.nodes['B'], 0)
                    else:
                        return frame.findSimilar(frame.nodes['C'], 0)
                # if there were no identical nodes found then it means that we can check for other trasnformations

            ## we start by checking for horizontal refelction

            if frame.checkHorizontalReflection(frame.nodes['A'], frame.nodes['B']):
                return frame.findSimilar(frame.createHorizonatalRefNode(frame.nodes['C']), 0)
            elif frame.checkHorizontalReflection(frame.nodes['A'], frame.nodes['C']):
                return frame.findSimilar(frame.createHorizonatalRefNode(frame.nodes['B']), 0)
            elif frame.checkHorizontalReflection(frame.nodes['B'], frame.nodes['C']):
                return frame.findSimilar(frame.createHorizonatalRefNode(frame.nodes['A']), 0)

            # we then check for vertical reflection
            if frame.checkVerticalReflection(frame.nodes['A'], frame.nodes['B']):
                return frame.findSimilar(frame.createVerticalRefNode(frame.nodes['C']), 0)
            elif frame.checkVerticalReflection(frame.nodes['A'], frame.nodes['C']):
                return frame.findSimilar(frame.createVerticalRefNode(frame.nodes['B']), 0)
            elif frame.checkVerticalReflection(frame.nodes['B'], frame.nodes['C']):
                return frame.findSimilar(frame.createVerticalRefNode(frame.nodes['A']), 0)
        else:
            #porblem type is 3x3

            # create a confidence dictionary for each option possible
            #  Each possible method is going to vote one option and we will select the option with the maximum number of votes
            self.confidence = dict()
            for i in range(1,9):
                self.confidence[i] = 0

            #############################################################################
            #  Method 1: identical check

            # check if all three are identical across rows
            # print("yes")
            if frame.probablisticCompare(frame.nodes['A'], frame.nodes['B']) and frame.probablisticCompare(frame.nodes['B'], frame.nodes['C']):

                if frame.probablisticCompare(frame.nodes['D'], frame.nodes['E']) and frame.probablisticCompare(frame.nodes['E'], frame.nodes['F']):
                    if frame.probablisticCompare(frame.nodes['G'], frame.nodes['H']):
                        temp = frame.findSimilar(frame.nodes['H'], 0)
                        if temp != -1:
                            return temp

            # check if all three are identical across coloumns
            if frame.probablisticCompare(frame.nodes['A'], frame.nodes['D']) and frame.probablisticCompare(frame.nodes['D'], frame.nodes['G']):
                if frame.probablisticCompare(frame.nodes['B'], frame.nodes['E']) and frame.probablisticCompare(frame.nodes['E'], frame.nodes['H']):
                    if frame.probablisticCompare(frame.nodes['C'], frame.nodes['F']):
                        temp = frame.findSimilar(frame.nodes['F'], 0)
                        if temp!=-1:
                            return temp

            # check if all three are identical across diagonal
            if frame.probablisticCompare(frame.nodes['A'], frame.nodes['E']):
                temp =  frame.findSimilar(frame.nodes['E'], 2)
                if temp!=-1:
                    return temp

            #  if there is no identical then lets look at black difference using black ratio

            #############################################################################
            #  Method 2: black ratio check


            # calculate average difference between black ratio within matrix
            solutionSet = []

            RATIO_TOLERANCE = 0.02
            if frame.nodes["G"].blackRatio > frame.nodes["H"].blackRatio:
                GH_difference = frame.nodes["G"].blackRatio - frame.nodes["H"].blackRatio
                if frame.nodes["C"].blackRatio > frame.nodes["F"].blackRatio:
                    #  case where ratio is reducing from G to H and ratio is reducing from C to F, then F-temp and H-temp should both be positve
                    CF_difference = frame.nodes["C"].blackRatio - frame.nodes["F"].blackRatio
                    for i in range(8):
                        HOption_difference = frame.nodes["H"].blackRatio-frame.nodes[str(i+1)].blackRatio
                        FOption_difference = frame.nodes["F"].blackRatio-frame.nodes[str(i+1)].blackRatio
                        if HOption_difference<RATIO_TOLERANCE + GH_difference and HOption_difference> GH_difference-RATIO_TOLERANCE and FOption_difference<RATIO_TOLERANCE + CF_difference and FOption_difference> CF_difference- RATIO_TOLERANCE:
                            solutionSet.append(i+1)
                else:
                    CF_difference = frame.nodes["F"].blackRatio - frame.nodes["C"].blackRatio
                    for i in range(8):
                        HOption_difference = frame.nodes["H"].blackRatio-frame.nodes[str(i+1)].blackRatio
                        FOption_difference = frame.nodes[str(i+1)].blackRatio- frame.nodes["F"].blackRatio
                        if HOption_difference<RATIO_TOLERANCE + GH_difference and HOption_difference>GH_difference-RATIO_TOLERANCE and FOption_difference<RATIO_TOLERANCE + CF_difference and FOption_difference>CF_difference- RATIO_TOLERANCE:
                            solutionSet.append(i+1)


            else:
                GH_difference = frame.nodes["H"].blackRatio - frame.nodes["G"].blackRatio
                if frame.nodes["C"].blackRatio > frame.nodes["F"].blackRatio:
                    #  case where ratio is reducing from G to H and ratio is reducing from C to F, then F-temp and H-temp should both be positve
                    CF_difference = frame.nodes["C"].blackRatio - frame.nodes["F"].blackRatio
                    for i in range(8):
                        HOption_difference = frame.nodes[str(i+1)].blackRatio-frame.nodes["H"].blackRatio
                        FOption_difference = frame.nodes["F"].blackRatio-frame.nodes[str(i+1)].blackRatio
                        if HOption_difference<RATIO_TOLERANCE + GH_difference and HOption_difference>GH_difference-RATIO_TOLERANCE and FOption_difference<RATIO_TOLERANCE + CF_difference and FOption_difference>CF_difference- RATIO_TOLERANCE:
                            solutionSet.append(i+1)
                else:
                    CF_difference = frame.nodes["F"].blackRatio - frame.nodes["C"].blackRatio
                    for i in range(8):
                        HOption_difference = frame.nodes[str(i+1)].blackRatio-frame.nodes["H"].blackRatio
                        FOption_difference = frame.nodes[str(i+1)].blackRatio- frame.nodes["F"].blackRatio
                        if HOption_difference<RATIO_TOLERANCE + GH_difference and HOption_difference>GH_difference-RATIO_TOLERANCE and FOption_difference<RATIO_TOLERANCE + CF_difference and FOption_difference>CF_difference- RATIO_TOLERANCE:
                            solutionSet.append(i+1)
            

            votes = dict()
            if len(solutionSet) !=0:
                if len(solutionSet) == 1:
                    return solutionSet[0]
                else:
                    # find ratio closest to average of F and H ratio
                    FHAverage = (frame.nodes["H"].blackRatio)
                    smallestDiff = -1
                    solution = -1
                    for i in solutionSet:
                        temp = abs(frame.nodes[str(i)].blackRatio - FHAverage)
                        if smallestDiff == -1:
                            solution = i
                            smallestDiff = temp
                        elif temp<smallestDiff:
                            solution = i
                            smallestDiff = temp

                    # return solution
                    votes[solution] = 1

            #############################################################################
            #  Method 3: avergae black ratio 
            blackRatioRow1 = abs(frame.nodes["A"].blackRatio + frame.nodes["B"].blackRatio + frame.nodes["C"].blackRatio)
            blackRatioRow2 = abs(frame.nodes["D"].blackRatio + frame.nodes["E"].blackRatio + frame.nodes["F"].blackRatio)
            blackRatioRow3 = abs(frame.nodes["G"].blackRatio + frame.nodes["H"].blackRatio)
            if abs(blackRatioRow1 - blackRatioRow2) < RATIO_TOLERANCE:
                for item in frame.nodes.keys():
                    if item not in {'A','B','C','D','E','F','G','H'}:
                        temp = blackRatioRow3 + frame.nodes[item].blackRatio
                        if abs(temp-blackRatioRow2) <RATIO_TOLERANCE:
                            if item not in votes.keys():
                                votes[item] = 1
                            else:
                                votes[item] += 1


            #############################################################################
            #  Method 4: add first two images of the row and check w the third one
            

            ### addition of first two images row-wise
            tempAB = Node(problem.figures["A"].visualFilename)

            tempAB.createNode(frame.addImage(frame.nodes["A"], frame.nodes["B"]))

            if frame.probablisticCompareBinary(tempAB, frame.nodes["C"]):
                # print("Here")
                tempDE = Node(problem.figures["A"].visualFilename)
                tempDE.createNode(frame.addImage(frame.nodes["D"], frame.nodes["E"]))
                if frame.probablisticCompareBinary(tempDE, frame.nodes["F"]):
                    # then the answer should be the option that is closest to the addisiotn of the G and H
                    temp =  Node(problem.figures["A"].visualFilename)
                    temp.createNode(frame.addImage(frame.nodes["G"], frame.nodes["H"]))

                    s = frame.findSimilar(temp,2)
                    if s not in votes.keys():
                        votes[s] = 3
                    else:
                        votes[s] +=3

            ### subtrcation of first two images column-wise
            tempAB = Node(problem.figures["A"].visualFilename)

            tempAB.createNode(frame.subtractImage(frame.nodes["D"], frame.nodes["A"]))

            if frame.probablisticCompareBinary(tempAB, frame.nodes["G"]):
                tempDE = Node(problem.figures["A"].visualFilename)
                tempDE.createNode(frame.subtractImage(frame.nodes["E"], frame.nodes["B"]))
                if frame.probablisticCompareBinary(tempDE, frame.nodes["H"]):
                    # then the answer should be the option that is closest to the addisiotn of the G and H
                    temp =  Node(problem.figures["A"].visualFilename)
                    temp.createNode(frame.subtractImage(frame.nodes["F"], frame.nodes["C"]))
                    s = frame.findSimilar(temp,2)
                    if s not in votes.keys():
                        votes[s] = 2
                    else:
                        votes[s] +=2


            ## select answer with maximum votes
            if len(votes.keys()) != 0:
                maximum = 0
                result = ""
                for opt in votes.keys():
                    if votes[opt] > maximum:
                        maximum = votes[opt]
                        result = opt
                # print(votes)
                return int(result)


# attempt at contour recognition
            # for node in frame.nodes.keys():
            #     # print(node)
            #     cnts = cv2.findContours(frame.nodes[node].binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            #     for cnt in cnts:
            #         approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            #         frame.nodes[node]
            #         print(node, ":", len(approx))




        # print("#################")
        return 5
