package naivebayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Basic Naive Bayes with Laplacian smoothing
 * 
 * @author amangrover
 *
 */
public class NaiveBayes {

	private Map<String, Map<String, Integer>> likelihood = new HashMap<String, Map<String, Integer>>();
	private Integer NUM_OF_ATTRIBUTE = 0;
	private Integer positiveCount = 0;
	private Integer negativeCount = 0;
	private Integer truePositive = 0;
	private Integer trueNegative = 0;
	private Integer falseNegative = 0;
	private Integer falsePositive = 0;

	private void train(String file) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		likelihood.put("+1", new HashMap<String, Integer>());
		likelihood.put("-1", new HashMap<String, Integer>());
		while ((line = br.readLine()) != null) {
			String[] wordIds = line.split("\\s+");
			String classLabel = wordIds[0];
			if (classLabel.equals("+1")) {
				positiveCount += 1;
			} else {
				negativeCount += 1;
			}
			for (int i = 1; i < wordIds.length; i++) {
				String keyVal = wordIds[i];
				Map<String, Integer> map = likelihood.get(classLabel);
				Integer val = map.get(keyVal);
				if (val == null) {
					val = 0;
				}
				map.put(keyVal, val + 1);
				if (Integer.parseInt(wordIds[i].split(":")[0]) > NUM_OF_ATTRIBUTE) {
					NUM_OF_ATTRIBUTE = Integer
							.parseInt(wordIds[i].split(":")[0]);
				}
			}
		}
	}

	private void test(String file) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		updateMax(file);
		double p = (double) positiveCount / (positiveCount + negativeCount);
		double n = (double) negativeCount / (positiveCount + negativeCount);
		while ((line = br.readLine()) != null) {
			String[] wordIds = line.split("\\s+");
			String originalClassLabel = wordIds[0];
			Double prob1 = 0.0;// for +1
			Double prob2 = 0.0;// for -1
			for (int i = 1; i < wordIds.length; i++) {
				String keyVal = wordIds[i];
				Map<String, Integer> mapPositive = likelihood.get("+1");
				Map<String, Integer> mapNegative = likelihood.get("-1");
				Integer count1 = mapPositive.get(wordIds[i]);
				Integer count2 = mapNegative.get(wordIds[i]);
				if (count1 == null) {
					count1 = 0;
				}
				if (count2 == null) {
					count2 = 0;
				}
				prob1 += Math.log((double) (count1 + 1)
						/ (positiveCount + NUM_OF_ATTRIBUTE));
				prob2 += Math.log((double) (count2 + 1)
						/ (negativeCount + NUM_OF_ATTRIBUTE));

			}
			prob1 += Math.log(p);
			prob2 += Math.log(n);
			String predictedClassLabel = "";
			if (prob1 < prob2) {
				predictedClassLabel = "-1";
			} else {
				predictedClassLabel = "+1";
			}

			if (predictedClassLabel.equals(originalClassLabel)) {
				if (predictedClassLabel.equals("+1")) {
					truePositive++;
				} else {
					trueNegative++;
				}
			} else {
				if (predictedClassLabel.equals("+1")) {
					falsePositive++;
				} else {
					falseNegative++;
				}
			}

		}

	}

	private void updateMax(String file) throws NumberFormatException,
			IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while ((line = br.readLine()) != null) {
			String[] wordIds = line.split("\\s+");
			String classLabel = wordIds[0];
			for (int i = 1; i < wordIds.length; i++) {
				String[] attrValue = wordIds[i].split(":");
				String attribute = attrValue[0];
				if (Integer.parseInt(attribute) > NUM_OF_ATTRIBUTE) {
					NUM_OF_ATTRIBUTE = Integer.parseInt(attribute);
				}
			}
		}
	}

	public void run(String train, String test) throws IOException {
		train(train);
		test(train);
		System.out.println(truePositive + " " + falseNegative + " "
				+ falsePositive + " " + trueNegative);
		Double accuracy = (double) (trueNegative + truePositive)
				/ (trueNegative + truePositive + falseNegative + falsePositive);
		System.out.println(accuracy);
		System.out.println("Error rate " + (1.0 - accuracy));
		System.out.println("Sensitivity "
				+ ((double) truePositive / (truePositive + falseNegative)));
		System.out.println("Specifity "
				+ ((double) trueNegative / (falsePositive + trueNegative)));
		Double precision = (double) truePositive
				/ (truePositive + falsePositive);
		System.out.println("Precision " + precision);
		Double recall = (double) truePositive / (truePositive + falseNegative);
		System.out.println("Recall " + recall);
		Double fMeasure = (2.0 * precision * recall) / (precision + recall);
		System.out.println("Fmeasure " + fMeasure);
		Double Fpoint2 = ((1 + 0.2 * 0.2) * precision * recall)
				/ (0.2 * 0.2 * precision + recall);
		System.out.println("F point 2 " + Fpoint2);
		Double F2 = ((1 + 2 * 2) * precision * recall)
				/ (2 * 2 * precision + recall);
		System.out.println("F2 " + F2);
		intialize();
		test(test);
		System.out.println();
		System.out.println(truePositive + " " + falseNegative + " "
				+ falsePositive + " " + trueNegative);
		Double accuracys = (double) (trueNegative + truePositive)
				/ (trueNegative + truePositive + falseNegative + falsePositive);
		System.out.println(accuracys);
		System.out.println("Error rate " + (1.0 - accuracy));
		System.out.println("Sensitivity "
				+ ((double) truePositive / (truePositive + falseNegative)));
		System.out.println("Specifity "
				+ ((double) trueNegative / (falsePositive + trueNegative)));
		precision = (double) truePositive / (truePositive + falsePositive);
		System.out.println("Precision " + precision);
		recall = (double) truePositive / (truePositive + falseNegative);
		System.out.println("Recall " + recall);
		fMeasure = (2.0 * precision * recall) / (precision + recall);
		System.out.println("Fmeasure " + fMeasure);
		Fpoint2 = ((1 + 0.2 * 0.2) * precision * recall)
				/ (0.2 * 0.2 * precision + recall);
		System.out.println("F point 2 " + Fpoint2);
		F2 = ((1 + 2 * 2) * precision * recall) / (2 * 2 * precision + recall);
		System.out.println("F2 " + F2);
	}

	private void intialize() {
		trueNegative = 0;
		truePositive = 0;
		falseNegative = 0;
		falsePositive = 0;

	}

	public static void main(String[] args) throws IOException {
		NaiveBayes bayes = new NaiveBayes();
		bayes.run(args[0], args[1]);
	}

}
