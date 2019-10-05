import java.io.*;

public class NaiveBayesClassifier {
	public static void main(String[] args) throws IOException {
		// reading training data file
		int i = 0, j = 0;
		int face = 0, f = 0;
		// declaring array containing information about pixels of each face
		char ft[][] = new char[451][4200];
		String sCurrentLine;

		BufferedReader br = new BufferedReader(new FileReader("E:/3-1/machine learning/ML Assignment/facedatatrain"));
		while ((sCurrentLine = br.readLine()) != null) {
			if (f == 4200) {
				f = 0;
				face++;
			}
			// System.out.println(f);
			for (j = 0; j < sCurrentLine.length(); j++) {
				ft[face][f] = sCurrentLine.charAt(j);
				f++;
			}
		}
		br.close();
		/*
		 * for(i=0;i<451;i++) { for(j=0;j<4200;j++) {
		 * System.out.println(ft[i][j]+" " + j + "  " + i); }
		 * System.out.println(); }
		 */
		// reading label file to calculate probability of being a face and not
		// being a face
		BufferedReader br1 = new BufferedReader(
				new FileReader("E:/3-1/machine learning/ML Assignment/facedatatrainlabels"));
		int train[] = new int[451];
		int facecount = 0, nonfacecount = 0;
		float pf, pnf;
		// setting of smoothconstant
		float smoothk = 1f;
		for (i = 0; i < 451; i++) {
			if (br1.read() == '0') {
				train[i] = 0;
				nonfacecount++;
			} else {
				train[i] = 1;
				facecount++;
			}
			br1.readLine();
		}
		br1.close();
		// printing face test labels training data
		/*
		 * for(i=0;i<451;i++) { System.out.println(train[i]); }
		 */
		// System.out.println(facecount+" "+nonfacecount);
		pf = (float) facecount / (facecount + nonfacecount);
		pnf = (float) nonfacecount / (facecount + nonfacecount);
		// System.out.println(pf+" "+pnf);

		// now calculating conditional probabilities

		float condhashfaceprob[] = new float[4200];
		float condspacefaceprob[] = new float[4200];
		int hashcount = 0, spacecount = 0;
		for (i = 0; i < 4200; i++) {
			hashcount = 0;
			spacecount = 0;
			for (j = 0; j < 451; j++) {
				if (ft[j][i] == '#' && train[j] == 1) {
					hashcount++;
				} else if (ft[j][i] == ' ' && train[j] == 1)
					spacecount++;
			}
			condhashfaceprob[i] = (hashcount + smoothk) / (hashcount + smoothk + spacecount + smoothk);
			condspacefaceprob[i] = (spacecount + smoothk) / (hashcount + smoothk + spacecount + smoothk);
		}

		float condhashnonprob[] = new float[4200];
		float condspacenonprob[] = new float[4200];

		for (i = 0; i < 4200; i++) {
			hashcount = 0;
			spacecount = 0;
			for (j = 0; j < 451; j++) {
				if (ft[j][i] == '#' && train[j] == 0) {
					hashcount++;
				} else if (ft[j][i] == ' ' && train[j] == 0)
					spacecount++;
			}
			condhashnonprob[i] = (float) (hashcount + smoothk) / (hashcount + smoothk + spacecount + smoothk);
			condspacenonprob[i] = (float) (spacecount + smoothk) / (hashcount + smoothk + spacecount + smoothk);
		}

		// for testing data

		// reading data from testing file
		i = 0;
		j = 0;
		face = 0;
		f = 0;
		char ftest[][] = new char[150][4200];

		BufferedReader br2 = new BufferedReader(new FileReader("E:/3-1/machine learning/ML Assignment/facedatatest"));
		while ((sCurrentLine = br2.readLine()) != null) {
			if (f == 4200) {
				f = 0;
				face++;
			}
			// System.out.println(f);
			for (j = 0; j < sCurrentLine.length(); j++) {
				ftest[face][f] = sCurrentLine.charAt(j);
				f++;
			}
		}
		br2.close();
		/*
		 * for(i=0;i<150;i++) { for(j=0;j<4200;j++) {
		 * System.out.println(ftest[i][j]+" " + j + "  " + i); }
		 * System.out.println(); }
		 * 
		 */

		// test labels
		BufferedReader br3 = new BufferedReader(
				new FileReader("E:/3-1/machine learning/ML Assignment/facedatatestlabels"));
		int actualOutput[] = new int[150];
		int predictedOutput[] = new int[150];
		int tn = 0, fp = 0, fn = 0, tp = 0;
		for (i = 0; i < 150; i++) {
			if (br3.read() == '0') {
				actualOutput[i] = 0;
			} else {
				actualOutput[i] = 1;

			}
			br3.readLine();
		}

		br3.close();
		// printing of test labels
		/*
		 * for(i=0;i<150;i++) { System.out.println(actualOutput[i]); }
		 */
		// confusion matrix
		double max1[] = new double[150];
		double max2[] = new double[150];
		for (i = 0; i < 150; i++) {
			max1[i] = Math.log(pf);
			for (j = 0; j < 4200; j++) {
				if (ftest[i][j] == ' ') {
					max1[i] = max1[i] + Math.log(condspacefaceprob[j]);
				} else if (ftest[i][j] == '#') {
					max1[i] = max1[i] + Math.log(condhashfaceprob[j]);
				}
			}
		}
		for (i = 0; i < 150; i++) {
			max2[i] = Math.log(pnf);
			for (j = 0; j < 4200; j++) {
				if (ftest[i][j] == ' ') {
					max2[i] = max2[i] + Math.log(condspacenonprob[j]);
				} else if (ftest[i][j] == '#') {
					max2[i] = max2[i] + Math.log(condhashnonprob[j]);
				}
			}
		}
		for (i = 0; i < 150; i++) {
			if (max2[i] > max1[i]) {
				predictedOutput[i] = 0;
			} else
				predictedOutput[i] = 1;
		}

		for (i = 0; i < 150; i++) {
			if (actualOutput[i] == 0 && predictedOutput[i] == 0) {
				tn++;
			} else if (actualOutput[i] == 0 && predictedOutput[i] == 1) {
				fp++;
				System.out.println("false positive occuring at i= " + i);
			} else if (actualOutput[i] == 1 && predictedOutput[i] == 0) {
				fn++;
				System.out.println("false negative occuring at i= " + i);
			} else if (actualOutput[i] == 1 && predictedOutput[i] == 1) {
				tp++;
			}
		}
		System.out.println(
				"false positive " + fp + " true positive " + tp + " false negative " + fn + " true negative" + tn);
		double accuracy = (double) (tp + tn) / (double) (fp + tp + fn + tn);
		System.out.println("smoothing constant " + smoothk + " accuracy " + accuracy);

		for (j = 0; j < 4200; j++) {
			
			System.out.print(ftest[16][j]);
			if(j+1 % 60 ==0)
			{
				System.out.println();
			}
		}

	}
}
