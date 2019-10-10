package algorithm;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@Slf4j
public class NaiveBayesClassifier {


    private int[] readActualOutput(String fileName) throws IOException {
        int[] actualOutput = new int[150];
        BufferedReader br3 = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(fileName)));
        for (int i = 0; i < 150; i++) {
            if (br3.read() == '0') {
                actualOutput[i] = 0;
            } else {
                actualOutput[i] = 1;

            }
            br3.readLine();
        }

        br3.close();
        return actualOutput;
    }

    private int[] readTrainingDataOutput(String fileName) throws IOException {
        BufferedReader br1 = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(fileName)));
        int[] train = new int[451];
        for (int i = 0; i < 451; i++) {
            if (br1.read() == '0') {
                train[i] = 0;

            } else {
                train[i] = 1;

            }
            br1.readLine();
        }
        br1.close();
        return train;
    }

    private int[] getPredictedOutput(float pf, float pnf, char[][] fTest, float[] condSpaceFaceProb,
                                     float[] condHashFaceProb, float[] condSpaceNonProb, float[] condHashNonProb) {
        int[] predictedOutput = new int[150];
        double[] max1 = new double[150];
        double[] max2 = new double[150];
        for (int i = 0; i < 150; i++) {
            max1[i] = Math.log(pf);
            for (int j = 0; j < 4200; j++) {
                if (fTest[i][j] == ' ') {
                    max1[i] = max1[i] + Math.log(condSpaceFaceProb[j]);
                } else if (fTest[i][j] == '#') {
                    max1[i] = max1[i] + Math.log(condHashFaceProb[j]);
                }
            }
        }
        for (int i = 0; i < 150; i++) {
            max2[i] = Math.log(pnf);
            for (int j = 0; j < 4200; j++) {
                if (fTest[i][j] == ' ') {
                    max2[i] = max2[i] + Math.log(condSpaceNonProb[j]);
                } else if (fTest[i][j] == '#') {
                    max2[i] = max2[i] + Math.log(condHashNonProb[j]);
                }
            }
        }
        for (int i = 0; i < 150; i++) {
            if (max2[i] > max1[i]) {
                predictedOutput[i] = 0;
            } else {
                predictedOutput[i] = 1;
            }

        }
        return predictedOutput;
    }


    private char[][] readFaceData(String fileName) throws IOException {
        int face = 0;
        int f = 0;
        char [][] faceData = new char[451][4200];
        String sCurrentLine;
        BufferedReader br = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(fileName)));
        while ((sCurrentLine = br.readLine()) != null) {
            if (f == 4200) {
                f = 0;
                face++;
            }

            for (int j = 0; j < sCurrentLine.length(); j++) {
                faceData[face][f] = sCurrentLine.charAt(j);
                f++;
            }
        }
        br.close();
        return  faceData;
    }

    private void printConfusionMatrixAndAccuracy(int[] actualOutput, int[] predictedOutput, float smoothk) {
        int i;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        int tp = 0;
        for (i = 0; i < 150; i++) {
            if (actualOutput[i] == 0 && predictedOutput[i] == 0) {
                tn++;
            } else if (actualOutput[i] == 0 && predictedOutput[i] == 1) {
                fp++;
                log.info("false positive occurring at i= " + i);
            } else if (actualOutput[i] == 1 && predictedOutput[i] == 0) {
                fn++;
                log.info("false negative occurring at i= " + i);
            } else if (actualOutput[i] == 1 && predictedOutput[i] == 1) {
                tp++;
            }
        }
        log.info(
                "false positive " + fp + " true positive " + tp + " false negative " + fn + " true negative" + tn);
        double accuracy = (double) (tp + tn) / (double) (fp + tp + fn + tn);
        log.info("smoothing constant " + smoothk + " accuracy " + accuracy);
    }


    void detectFace() throws IOException {

        // setting of smoothconstant
        float smoothk = 1f;

        // reading training data file
        int i;
        int j;
        // declaring array containing information about pixels of each face
        char[][] ft = readFaceData("/faceDataTrain");

        // reading label file to calculate probability of being a face and not being a face
        int[] train = readTrainingDataOutput("/faceDataTrainLabels");
        int faceCount = 0;
        int nonFaceCount = 0;
        float pf;
        float pnf;


        for(i=0; i< 451; i++) {
            if(train[i] == 0) {
                nonFaceCount++;
            }
            else {
                faceCount++;
            }
        }

        pf = (float) faceCount / (faceCount + nonFaceCount);
        pnf = (float) nonFaceCount / (faceCount + nonFaceCount);

        // now calculating conditional probabilities
        float[] condHashFaceProb = new float[4200];
        float[] condSpaceFaceProb = new float[4200];
        int hashCount;
        int spaceCount;
        for (i = 0; i < 4200; i++) {
            hashCount = 0;
            spaceCount = 0;
            for (j = 0; j < 451; j++) {
                if (ft[j][i] == '#' && train[j] == 1) {
                    hashCount++;
                } else if (ft[j][i] == ' ' && train[j] == 1) {
                    spaceCount++;
                }

            }
            condHashFaceProb[i] = (hashCount + smoothk) / (hashCount + smoothk + spaceCount + smoothk);
            condSpaceFaceProb[i] = (spaceCount + smoothk) / (hashCount + smoothk + spaceCount + smoothk);
        }

        float[] condHashNonProb = new float[4200];
        float[] condSpaceNonProb = new float[4200];

        for (i = 0; i < 4200; i++) {
            hashCount = 0;
            spaceCount = 0;
            for (j = 0; j < 451; j++) {
                if (ft[j][i] == '#' && train[j] == 0) {
                    hashCount++;
                } else if (ft[j][i] == ' ' && train[j] == 0) {
                    spaceCount++;
                }
            }
            condHashNonProb[i] = (hashCount + smoothk) / (hashCount + smoothk + spaceCount + smoothk);
            condSpaceNonProb[i] = (spaceCount + smoothk) / (hashCount + smoothk + spaceCount + smoothk);
        }

        // for testing data

        // reading data from testing file
        char[][] fTest = readFaceData("/faceDataTest");
        // test labels
        int[] actualOutput = readActualOutput("/faceDataTestLabels");
        int[] predictedOutput = getPredictedOutput(pf, pnf, fTest, condSpaceFaceProb, condHashFaceProb,
                condSpaceNonProb, condHashNonProb);

        // confusion matrix
        printConfusionMatrixAndAccuracy(actualOutput, predictedOutput, smoothk);

    }

}
