import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# globally loaded sign names csv data
gTrafficSignClassifierCsvData = pd.read_csv("signnames.csv")

'''
    Helper for plotting multiple images and giving them titles
'''

def plotImages(images, titles=[""], columns=1, figsize=(20,10), gray=False, saveAs=''):
    errorStr = "plotImages failed..."
    # images and titles must be lists
    if(not isinstance(images, (list,)) or not isinstance(titles, (list,))):
        print(errorStr + " images/titles are not both instances of list")
        return
    
    # the number of titles must match the number of columns OR
    # match the number of images
    if(len(titles) != columns and len(titles) != len(images)):
        print(errorStr + " images/titles are not the same length")
        return
    
    plt.figure(figsize=figsize)
    
    fig = plt.gcf()
    
    for i, image in enumerate(images):
        rows = math.ceil(len(images) / columns)
        plt.subplot(rows, columns, i + 1)
        
        if len(images) == len(titles):
            plt.gca().set_title(titles[i])
        else:
            plt.gca().set_title(titles[i % columns])
       
        # if gray is a list, each item  
        # corresponds to if each row is gray
        tmpGray = gray
        if isinstance(gray, (list,)):
            tmpGray = gray[i // columns]
            
        if gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

    if saveAs != '':
        fig.savefig(saveAs, dpi=fig.dpi)

'''
    Helper for getting the traffic sign name from an integer id
'''
        
def trafficSignName(classifierId):
    '''
        classifierId -> integer that corresponds to a traffic sign id in the lookup table
        
        returns a string name of the traffic sign if found and the string classifierId if not found
        
        Note: uses global data loaded from traffic sign classifier id lookup table found in csv file
    '''
    
    global gTrafficSignClassifierCsvData
    
    data = gTrafficSignClassifierCsvData['SignName'].where(gTrafficSignClassifierCsvData['ClassId'] == classifierId).dropna()
    
    if len(data):
        return data.iloc(0)[0]
    
    return str(classifierId)