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
	private Map<String, Integer> attributesClass = new HashMap<String, Integer>();
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
		attributesClass.put("+1", 0);
		attributesClass.put("-1", 0);
		while ((line = br.readLine()) != null) {
			String[] wordIds = line.split("\\s+");
			String classLabel = wordIds[0];
			if (classLabel.equals("+1")) {
				positiveCount = positiveCount + 1;
			} else {
				negativeCount += 1;
			}
			for (int i = 1; i < wordIds.length; i++) {
				String[] attrValue = wordIds[i].split(":");
				String attribute = attrValue[0];
				Integer value = Integer.parseInt(attrValue[1]);
				Map<String, Integer> map = likelihood.get(classLabel);
				Integer val = map.get(attribute);
				if (val == null) {
					val = value;
				} else {
					val = val + value;
				}
				map.put(attribute, val);
				attributesClass.put(classLabel, attributesClass.get(classLabel)
						+ value);
				if (Integer.parseInt(attribute) > NUM_OF_ATTRIBUTE) {
					NUM_OF_ATTRIBUTE = Integer.parseInt(attribute);
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
			if(originalClassLabel.equals("+1")){
				int a=1;
			}
			Double prob1 = 0.0;// for +1
			Double prob2 = 0.0;// for -1
			for (int i = 1; i < wordIds.length; i++) {
				String[] attrValue = wordIds[i].split(":");
				String attribute = attrValue[0];
				Integer value = Integer.parseInt(attrValue[1]);
				Integer positive = likelihood.get("+1").get(attribute);
				Integer negative = likelihood.get("-1").get(attribute);
				if (positive == null) {
					positive = 0;
				}
				if (negative == null) {
					negative = 0;
				}
				prob1 += Math.log((double) (positive + 1)
						/ (attributesClass.get("+1") + NUM_OF_ATTRIBUTE));
				prob2 += Math.log((double) (negative + 1)
						/ (attributesClass.get("+1") + NUM_OF_ATTRIBUTE));

			}
			prob1 += Math.log(p);
			prob2 += Math.log(n);
			String predictedClassLabel = "";
			if (prob1 > prob2) {
				predictedClassLabel = "+1";
			} else {
				predictedClassLabel = "-1";
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
		System.out.print(truePositive + " " + falseNegative + " "
				+ falsePositive + " " + trueNegative);
		intialize();
		test(test);
		System.out.print(truePositive + " " + falseNegative + " "
				+ falsePositive + " " + trueNegative);
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
