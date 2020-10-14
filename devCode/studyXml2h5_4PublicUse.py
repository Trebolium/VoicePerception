import os, h5py
from xml.etree import ElementTree
import util
import numpy as np

parent_folder_path = '/Users/brendanoconnor/Desktop/phd/ListeningTest/vocal_perception_study_small_version/phpAssignXml/onlineStudy/savesFromServer/actualSaves-20200629'
h5_path_destination = '/Users/brendanoconnor/Desktop/anonomisedData.hdf5'

# remember to check out h5pyViewer
if 'h5' in globals():
    h5.close()

h5 = h5py.File(h5_path_destination, mode='w')
dt = h5py.special_dtype(vlen=str)

label_list = ['straight','belt','breathy','fry','vibrato']
standard_ref_seq = ['straight','straight','straight','belt','belt','belt','breathy','breathy','breathy','fry','fry','fry','vibrato','vibrato','vibrato']
pageOffset = 5
audioElementOffset = 3
totalPages = 20
practisePages = 3
infoPerPage = 3
audiosPerRating = 2
numberOfParticipants = 0
perfect_msi_score=9*7
filePathList = []

# go through parent path. Count and save a list of files that are complete
for root, dir, files in os.walk(parent_folder_path):
    for file in files:
        # if save file is completed, count it
        if file.endswith('.xml'):
            file_path = root + '/' + file
            myXmlTree = ElementTree.parse(file_path)
            xmlRoot = myXmlTree.getroot()
            for page in xmlRoot.findall('page'):
                if page.attrib['presentedId']=="19":
                    if page.attrib['state']=="complete":
                        filePathList.append(file_path)
                        numberOfParticipants += 1

# note: when updating hdf5 file, you must put the entirety of the entry in for the index of the 0th dimension

# filename, session, time, date, equipment, environment, age, gender
# MSI-singingJudgement, MSI-hearingFirstTime, MSI-spotMistakes, MSI-compareVersions
# MSI-songFamiliarity, MSI-beatPerceptions, MSI-perceiveTuning, MSI-selfTunePerception, MSI-genreIdentity
# bestInstrument, selfReportedSimilarityJudgement, hearingImpairments, timeTaken, msi-score, musicianCategory, selfReport, reliability
infoPerParticipant = 29
totalPages4Data = 14
ratingsPerPage = 8
totalRepeatedPages = 2

h5.create_dataset('participantInfo',
                  shape=(numberOfParticipants, infoPerParticipant),
                  dtype=dt)
h5.create_dataset('pageInfo',
                  shape=(numberOfParticipants, totalPages4Data, infoPerPage),
                  dtype=dt)

h5.create_dataset('referenceAudioNames',
                  shape=(numberOfParticipants, totalPages4Data),
                  dtype=dt)
h5.create_dataset('rearrangedReferenceAudioNames',
                  shape=(numberOfParticipants, totalPages4Data),
                  dtype=dt)
h5.create_dataset('comparativeAudioNames',
                  shape=(numberOfParticipants, totalPages4Data, ratingsPerPage),
                  dtype=dt)
h5.create_dataset('participantRatings',
                  shape=(numberOfParticipants, totalPages4Data, ratingsPerPage),
                  dtype=np.float)

h5.create_dataset('repeatReferenceAudioNames',
                  shape=(numberOfParticipants, totalRepeatedPages),
                  dtype=dt)
h5.create_dataset('repeatComparativeAudioNames',
                  shape=(numberOfParticipants, totalRepeatedPages, ratingsPerPage),
                  dtype=dt)
h5.create_dataset('repeatParticipantRatings',
                  shape=(numberOfParticipants, totalRepeatedPages, ratingsPerPage),
                  dtype=np.float)

# then entries in this matrix from 0-n will be of the same order as 
h5.create_dataset('dissimMatrix',
                  shape=(numberOfParticipants, totalPages4Data, totalPages4Data),
                  dtype=np.float)
h5.create_dataset('rearrangedDissimMatrix',
                  shape=(numberOfParticipants, totalPages4Data, totalPages4Data),
                  dtype=np.float)
h5.create_dataset('dissimMatrix15Dims',
                  shape=(numberOfParticipants, totalPages4Data+1, totalPages4Data+1),
                  dtype=np.float)

h5.create_dataset('mds2d',
                  shape=(numberOfParticipants, totalPages4Data, 2),
                  dtype=np.float)
h5.create_dataset('mds3d',
                  shape=(numberOfParticipants, totalPages4Data, 3),
                  dtype=np.float)

def xmlRooter (file_path):
    myXmlTree = ElementTree.parse(file_path)
    xmlRoot = myXmlTree.getroot()
    return xmlRoot

experiment_idx = 0
# look at all xml files
for file_path in filePathList:
    print('Processing Participate {0}'.format(experiment_idx))
    xmlRoot = xmlRooter(file_path)
    participantDetails = []
    saveName = xmlRoot.attrib['key']
    participantDetails.append('censoredSaveInfo')
    _, singerSessionId = xmlRoot[0].attrib['url'].rsplit('/', 1)
    participantDetails.append(singerSessionId[:-4])
    date = xmlRoot[1][0].attrib['day'] + ':' + xmlRoot[1][0].attrib['month'] + ':' + xmlRoot[1][0].attrib['year']
    participantDetails.append(date)
    time = xmlRoot[1][1].attrib['hour'] + ':' + xmlRoot[1][1].attrib['minute'] + ':' + xmlRoot[1][1].attrib['secs']
    participantDetails.append(time)

    # CURRENTLY I DON[T CARE ABOUT THIS INFO - ITS CAUSING HASSLE AND NOT USEFUL DATA, YET!
    # navigatorResults = xmlRoot[2]
    # for idx, navigator in enumerate(navigatorResults):
    #     if navigator.text != None:
    #         # print('navigatorResults: ', idx, navigator.text)
    #         participantDetails.append(navigator.text)

    # collect information from the survey
    preSurveyResults = xmlRoot[3]
    for idx, surveyresult in enumerate(preSurveyResults):
        if idx == 0:
            contactInfo = surveyresult[0].text
            participantDetails.append('censoredContactInfo')
        elif idx == 3:
            # idx 7 is age
            age = int(surveyresult[0].text)
            participantDetails.append(age)
        elif idx == 4:
            # genderId = int(surveyresult[0].text) # change the entries manually to numerical categories
            # idx 8 is gender
            genderId = surveyresult[0].text
            participantDetails.append(genderId)
        elif idx == 14:
            instrument = surveyresult[0].text
            participantDetails.append(instrument)
        else:
            # idx 5-6 are listening gear and environment
            # idx 9-17
            surveyChoiceResult = surveyresult[0].attrib['name']
            participantDetails.append(surveyChoiceResult)
    selfReportedSimilarityJudgement = xmlRoot[4][0][0].text
    participantDetails.append(selfReportedSimilarityJudgement)

    #prepare lists for populating page and element information
    allRatingsPerPage=[]
    allInfoPerPage=[]
    allCompAudioPerPart=[]
    allRefAudioPerPart=[]
    #repeated data
    allRepeatRatingsPerPage=[]
    allRepeatCompAudioPerPart=[]
    allRepeatRefAudioPerPart=[]    
    
    # audio_paths=set()
    totalTimePerParticipant=0
    
    # go through pages in xml to collect infoPerPage and get lists for ref and comp audio
    audioRefSet=set()
    audioCompSet=set()
    for pageNum in range(pageOffset,
                         pageOffset + totalPages):  # skip tag 4, which is survey, and go through the 19 pages
        
        # prepare lists for populating page group information
        infoPerPage=[]

        # initiate xml page object and specific info
        page = xmlRoot[pageNum]
        realPageNum = pageNum - pageOffset
        pagePresentedId = int(page.attrib['presentedId'])
        infoPerPage.append(pagePresentedId-practisePages)
        
        # go through tags in metric and if their id says testTime, add their text value
        for metricTag in page[2].iter():
            if metricTag.get('id')=='testTime':
                pageDuration = metricTag.text
        totalTimePerParticipant += float(pageDuration)
        infoPerPage.append(pageDuration)

        # analyse ref names to decide how to treat pages. If they have continue, it means ignore
        if (page.attrib['ref'] == 'practisePage2') or (page.attrib['ref'] == 'practisePage0') or (page.attrib['ref'] == 'page13') or ('repeat' in page.attrib['ref']): #ignore this practise page
            continue
        elif page.attrib['ref'] == 'page1': # ignore practise page but take listeningImpairment results
            # participantDetails idx #20
            listeningImpairment = page[1][0][0].attrib['name']
            participantDetails.append(listeningImpairment)
            continue
        elif page.attrib['ref'] == 'practisePage1': #DON'T IGNORE! Fix label and proceed
            pageRef = '01'
        else: # for everything else, ensure pageRef is in page## format
            if len(page.attrib['ref'])==5:
                pageRef = '0' + page.attrib['ref'][4:]
            else:
                pageRef = page.attrib['ref'][4:]
        # now that ref is managed, add it to the page info list
        # WARNING: ONE PAGE IS MISSING FROM HERE AS IT IS THE NEGLECTED AUDIO PAGE
        infoPerPage.append(int(pageRef))
    
        allInfoPerPage.append(infoPerPage)
        
        # this first element loop is to collect the names of all audio files
        for idx, audioelement in enumerate(page):
            if idx < audioElementOffset:  # first 3 tags in page aren't audioelements
                continue

            realIdx = idx - audioElementOffset
            # metric = audioelement[0]
            if realIdx == 9:
                break
            if realIdx == 8:
                _, reference_audio_name = audioelement.attrib['fqurl'].rsplit('/',1)
                audioRefSet.add(reference_audio_name)
#                print('audioRefSet',audioRefSet)
            else:
                # print(realPageNum, realIdx)
                _, comparative_audio_name = audioelement.attrib['fqurl'].rsplit('/',1)
                audioCompSet.add(comparative_audio_name)
        
    # check if these names are in the refAudioList
    for name in audioCompSet:
        if name not in audioRefSet:
#            print(name, 'not in refSet')
            neglected_audio_name = name
#            print(name, 'in refset')

    # go through pages in xml
    for pageNum in range(pageOffset,
                         pageOffset + totalPages):  # skip tag 4, which is survey, and go through the 19 pages
        
        ratingsPerPage=[]
        audioPairsPerPage=[]
        comparative_audio_name_list=[]
        
        comparative_audio_name=''
        reference_audio_name=''

        page = xmlRoot[pageNum]
        realPageNum = pageNum - pageOffset

        # analyse ref names to decide how to treat pages. If they have continue, it means ignore
        if (page.attrib['ref'] == 'practisePage2') or (page.attrib['ref'] == 'practisePage0') or (page.attrib['ref'] == 'page13') or (page.attrib['ref'] == 'page1'): #ignore this practise page
            continue

        # go through elements in each page
        for idx, audioelement in enumerate(page):
            if idx < audioElementOffset:  # first 3 tags in page aren't audioelements
                continue

            realIdx = idx - audioElementOffset
            # metric = audioelement[0]
            if realIdx == 9:
                break
            if realIdx == 8:
                _, reference_audio_name = audioelement.attrib['fqurl'].rsplit('/',1)
            else:
                # print(realPageNum, realIdx)
                _, comparative_audio_name = audioelement.attrib['fqurl'].rsplit('/',1)
                if comparative_audio_name == neglected_audio_name:
#                    print(neglected_audio_name, 'comparison found at page: ', realPageNum, 'realIdx', realIdx)
                    value = audioelement[2].text
                    comparative_audio_name_list.append('neglected_audio_name')
                    ratingsPerPage.append(float(1 - float(value)))
                else:
                    value = audioelement[2].text
                    comparative_audio_name_list.append(comparative_audio_name)
                    # 1 - value so that polarisation is inversed, anything close to 0 becomes close to 1, 0 becomes 1, 1 becomes 0
                    ratingsPerPage.append(float(1 - float(value)))
                    
        if ('repeat' in page.attrib['ref']):
            [audioPairsPerPage.append(path) for path in comparative_audio_name_list]
            allRepeatCompAudioPerPart.append(audioPairsPerPage)
            # ref audio will have to stay in the same order for all
            allRepeatRefAudioPerPart.append(reference_audio_name)
            # print('allCompAudioPerPart: ', len(allCompAudioPerPart))
            allRepeatRatingsPerPage.append(ratingsPerPage)            
        else:
            [audioPairsPerPage.append(path) for path in comparative_audio_name_list]
            allCompAudioPerPart.append(audioPairsPerPage)
            # ref audio will have to stay in the same order for all
            allRefAudioPerPart.append(reference_audio_name)
            # print('allCompAudioPerPart: ', len(allCompAudioPerPart))
            allRatingsPerPage.append(ratingsPerPage)

    # idx 21
    participantDetails.append(totalTimePerParticipant)
    # participantInfo 22
    listening_factor = int(participantDetails[5]) + int(participantDetails[6])
    participantDetails.append(listening_factor)
    
    msi_score = int(participantDetails[9]) + int(participantDetails[10]) + int(participantDetails[11]) + int(participantDetails[12]) + int(participantDetails[13]) + int(participantDetails[14]) + int(participantDetails[15]) + int(participantDetails[16]) + int(participantDetails[17])    
    msi_percentage = msi_score*100/perfect_msi_score
    # participantInfo 23
    participantDetails.append(msi_percentage)
    
    musician_category = ''
    if participantDetails[18].startswith('n'):
        musician_category = '0'
    elif participantDetails[18].startswith('v'):
        musician_category = '2'
    else:
        musician_category = '1'
    # participantInfo 24
    participantDetails.append(musician_category)
    # participantInfo 25
    task_understood = participantDetails[19][:1]
    participantDetails.append(task_understood)
    
    h5['pageInfo'][experiment_idx] = allInfoPerPage
    
    h5['participantRatings'][experiment_idx] = allRatingsPerPage
    h5['referenceAudioNames'][experiment_idx] = allRefAudioPerPart
    h5['comparativeAudioNames'][experiment_idx] = allCompAudioPerPart
    
    h5['repeatParticipantRatings'][experiment_idx] = allRepeatRatingsPerPage
    h5['repeatReferenceAudioNames'][experiment_idx] = allRepeatRefAudioPerPart
    h5['repeatComparativeAudioNames'][experiment_idx] = allRepeatCompAudioPerPart
    
    h5['dissimMatrix'][experiment_idx] = util.generate_dissim_matrix(h5, experiment_idx)
    
    rearranged_ref_audio_names, target_indeces = util.arrange_indices_by_labels(h5, experiment_idx, h5['referenceAudioNames'][experiment_idx])
    rearranged_matrix = util.reorder_matrix(h5, h5['dissimMatrix'][experiment_idx], target_indeces)
    corrected_rearranged_matrix, zero_error = util.zeroed_dissim_matrix(rearranged_matrix, totalPages4Data)
    
    #participantInfo 26
    participantDetails.append(zero_error)
    
    h5['rearrangedDissimMatrix'][experiment_idx] = corrected_rearranged_matrix
    h5['rearrangedReferenceAudioNames'][experiment_idx] = rearranged_ref_audio_names
    
    h5['mds2d'][experiment_idx] = util.mds_from_dissim_matrix(h5['rearrangedDissimMatrix'][experiment_idx], h5['rearrangedReferenceAudioNames'][experiment_idx], 2)
    h5['mds3d'][experiment_idx] = util.mds_from_dissim_matrix(h5['rearrangedDissimMatrix'][experiment_idx], h5['rearrangedReferenceAudioNames'][experiment_idx], 3)
        
    page6, page14 = util.evaluate_reliabiltiy(h5, experiment_idx)
    if page14>=page6:
        reliability = str(page14)
    else:
        reliability = str(page6)
    #participantInfo 27
    participantDetails.append(reliability)
    #participantInfo 28
    participantDetails.append(neglected_audio_name +' neglected_audio_name')
    h5['participantInfo'][experiment_idx] = participantDetails
    h5['dissimMatrix15Dims'][experiment_idx] = util.generate_15dim_matrix(h5, experiment_idx, label_list)
    
    experiment_idx += 1

h5.close()

