import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests

datasetDir ="../data/"

verbose = False
showResults = False
showCED = True

normalization = 'corners'
failureThreshold = 0.08

networkFilename = "..\\DAN.npz"
network = FaceAlignment(112, 112, 1, nStages=2)
network.loadNetwork(networkFilename)

print ("Network being tested: " + networkFilename)
print ("Normalization is set to: " + normalization)
print ("Failure threshold is set to: " + str(failureThreshold))

commonSet = ImageServer.Load(datasetDir + "commonSet.npz")
challengingSet = ImageServer.Load(datasetDir + "challengingSet.npz")
w300 = ImageServer.Load(datasetDir + "w300Set.npz")

print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
commonErrs = tests.LandmarkError(commonSet, network, normalization, showResults, verbose)
print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
challengingErrs = tests.LandmarkError(challengingSet, network, normalization, showResults, verbose)

fullsetErrs = commonErrs + challengingErrs
print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN")
print("Average error: {0}".format(np.mean(fullsetErrs)))
tests.AUCError(fullsetErrs, failureThreshold, showCurve=showCED)

print ("Processing 300W private test set")
w300Errs = tests.LandmarkError(w300, network, normalization, showResults, verbose)
tests.AUCError(w300Errs, failureThreshold, showCurve=showCED)

