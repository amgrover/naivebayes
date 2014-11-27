package naivebayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Basic Naive Bayes with Laplacian smoothing
 * 
 * @author amangrover
 *
 */
public class NBAdaBoost {
	class Instance implements Comparable<Instance> {
		String instance;
		Double weight;

		public Instance(String in, Double we) {
			instance = in;
			weight = we;
		}

		@Override
		public int compareTo(Instance o) {
			return this.weight.compareTo(o.weight);
		}

	}

	class Error {
		Double error;
		Integer positiveCount;
		Integer negativeCount;

		public Error(Double er, Integer p, Integer n) {
			error = er;
			positiveCount = p;
			negativeCount = n;
		}
	}

	private Map<Map<String, Map<String, Integer>>, Error> classiffier = new HashMap<Map<String, Map<String, Integer>>, Error>();
	private Integer truePositive = 0;
	private Integer trueNegative = 0;
	private Integer falseNegative = 0;
	private Integer falsePositive = 0;
	private Integer max = 0;

	private void train(List<Instance> instances) throws IOException {
		Instance[] fulldata = instances.toArray(new Instance[instances.size()]);
		int NUM_OF_ATTRIBUTE = updateMax(instances);
		max = NUM_OF_ATTRIBUTE;
		int k = 5;
		while (k > 0) {
			Map<String, Map<String, Integer>> likelihood = new HashMap<String, Map<String, Integer>>();
			likelihood.put("+1", new HashMap<String, Integer>());
			likelihood.put("-1", new HashMap<String, Integer>());
			int size = (int) (fulldata.length * 0.80);
			Instance[] sampledData = getSample(size, fulldata);
			Integer positiveCount = 0;
			Integer negativeCount = 0;
			for (Instance instance : sampledData) {
				String line = instance.instance;
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
				}
			}

			List<Integer> correctlyClassified = new ArrayList<Integer>();
			Double error = testOnTrain(positiveCount, negativeCount,
					likelihood, correctlyClassified, fulldata, NUM_OF_ATTRIBUTE);
			if (error > 0.5) {
				continue;
			}
			classiffier.put(likelihood, new Error(error, positiveCount,
					negativeCount));
			double updated = error / (1 - error);
			Double oldWeight = 0.0;
			for (Instance instance : fulldata) {
				oldWeight += instance.weight;
			}
			for (Integer integer : correctlyClassified) {
				fulldata[integer].weight *= updated;
			}
			Double newWeightSum = 0.0;
			for (Instance instance : fulldata) {
				newWeightSum += instance.weight;
			}
			Double factor = oldWeight / newWeightSum;
			for (Instance instance : fulldata) {
				instance.weight *= factor;
			}
			k--;

		}

	}

	private Double testOnTrain(Integer positiveCount, Integer negativeCount,
			Map<String, Map<String, Integer>> likelihood,
			List<Integer> correctlyClassified, Instance[] fulldata,
			Integer NUM_OF_ATTRIBUTE) {
		Double error = 0.0;
		double p = (double) positiveCount / (positiveCount + negativeCount);
		double n = (double) negativeCount / (positiveCount + negativeCount);
		for (int m = 0; m < fulldata.length; m++) {
			String line = fulldata[m].instance;
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

			if (!predictedClassLabel.equals(originalClassLabel)) {
				error += fulldata[m].weight;
			} else {
				correctlyClassified.add(m);
			}

		}
		return error;
	}

	private void test(List<Instance> instances) throws IOException {
		int NUM_OF_ATTRIBUTE = updateMax(instances);
		NUM_OF_ATTRIBUTE = NUM_OF_ATTRIBUTE > max ? NUM_OF_ATTRIBUTE : max;
		for (Instance instance : instances) {
			String line = instance.instance;
			Double c1 = 0.0;
			Double c2 = 0.0;
			String[] wordIds = line.split("\\s+");
			String originalClassLabel = wordIds[0];
			for (Map.Entry<Map<String, Map<String, Integer>>, Error> entry : classiffier
					.entrySet()) {
				Map<String, Map<String, Integer>> likelihood = entry.getKey();
				Error error = entry.getValue();
				int positiveCount = error.positiveCount;
				int negativeCount = error.negativeCount;
				double p = (double) positiveCount
						/ (positiveCount + negativeCount);
				double n = (double) negativeCount
						/ (positiveCount + negativeCount);

				Double prob1 = 0.0;// for +1
				Double prob2 = 0.0;// for -1
				for (int i = 1; i < wordIds.length; i++) {
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
				Double we = Math.log((1 - error.error) / error.error);
				if (prob1 > prob2) {
					c1 += we;
				} else {
					c2 += we;
				}

			}
			String predictedClassLabel = "";
			if (c1 > c2) {
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

	private Integer updateMax(List<Instance> instances)
			throws NumberFormatException, IOException {
		int max = 0;
		for (Instance instance : instances) {
			String line = instance.instance;
			String[] wordIds = line.split("\\s+");
			String classLabel = wordIds[0];
			for (int i = 1; i < wordIds.length; i++) {
				String[] attrValue = wordIds[i].split(":");
				String attribute = attrValue[0];
				if (Integer.parseInt(attribute) > max) {
					max = Integer.parseInt(attribute);
				}
			}
		}
		return max;
	}

	public void run(String train, String test) throws IOException {
		List<Instance> instances = loadDataIntoMemory(train);
		train(instances);
		test(instances);
		System.out.println(truePositive + " " + falseNegative + " "
				+ falsePositive + " " + trueNegative);
		Double accuracy = (double) (trueNegative + truePositive)
				/ (trueNegative + truePositive + falseNegative + falsePositive);
		System.out.println(accuracy * 100);
		intialize();
		
		List<Instance> instances2 = loadDataIntoMemory(test);
		test(instances2);
		System.out.println();
		System.out.println(truePositive + " " + falseNegative + " "
				+ falsePositive + " " + trueNegative);
		Double accuracys = (double) (trueNegative + truePositive)
				/ (trueNegative + truePositive + falseNegative + falsePositive);
		System.out.println(accuracys * 100);
	}

	private void intialize() {
		trueNegative = 0;
		truePositive = 0;
		falseNegative = 0;
		falsePositive = 0;

	}

	private List<Instance> loadDataIntoMemory(String file)
			throws NumberFormatException, IOException {
		List<Instance> list = new ArrayList<Instance>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while ((line = br.readLine()) != null) {
			list.add(new Instance(line, 0.0));
		}
		for (Instance instance : list) {
			instance.weight = (double) 1 / list.size();
		}
		return list;
	}

	private Instance[] getSample(Integer size, Instance[] instances) {
		Arrays.sort(instances);
		Instance[] strings = new Instance[size];
		double[] array = new double[instances.length];
		array[0] = instances[0].weight;
		for (int i = 1; i < instances.length; i++) {
			array[i] = instances[i].weight + array[i - 1];
		}
		for (int i = 0; i < size; i++) {
			double num = Math.random();
			if (num < instances[0].weight) {
				strings[i] = new Instance(instances[0].instance,
						instances[0].weight);

			} else {
				int index = binarySearch(array, num);
				strings[i] = new Instance(instances[index].instance,
						instances[index].weight);
			}
		}
		return strings;

	}

	private int binarySearch(double[] array, double num) {
		int start = 0;
		int end = array.length;
		while (end - start > 1) {
			int mid = (start + end) / 2;
			if (array[mid] >= num) {
				end = mid;
			} else {
				start = mid;
			}
		}
		return end;
	}

	public static void main(String[] args) throws IOException {
		NBAdaBoost bayes = new NBAdaBoost();
		bayes.run(args[0], args[1]);
	}

}
